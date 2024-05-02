#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    pypi_db.py
    ~~~~~~~~~~

    PyPI package database

    :copyright: (c) 2013-2015 by Jauhien Piatlicki
    :license: GPL-2, see LICENSE for more details.
"""

import datetime
import enum
import json
import operator
import os
import pathlib
import re
import string
import subprocess
import tempfile
import zipfile
import shutil

from g_sorcery.exceptions import DownloadingError
from g_sorcery.fileutils import wget
from g_sorcery.g_collections import (
    Dependency, Package, serializable_elist, Version)
from g_sorcery.package_db import DBGenerator,PackageDB
from g_sorcery.logger import Logger

_logger = Logger()


PYTHON_VERSIONS = {Version((3, 10)), Version((3, 11)), Version((3, 12))}


def containment(fun):
    import functools

    @functools.wraps(fun)
    def newfun(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except:
            import traceback
            _logger.error(traceback.format_exc())
            _logger.error(f'ARGUMENTS {args=} {kwargs=}')

    return newfun


def pypi_normalize(pkg):
    return re.sub(r"[-_.]+", "-", pkg).lower()


class Operator(enum.Enum):
    LESS = 1
    LESSEQUAL = 2
    EQUAL = 3
    SIMILAR = 4
    GREATEREQUAL = 5
    GREATER = 6
    UNEQUAL = 7

    def compare(self, first, second):
        comparators = {
            Operator.LESS: operator.lt,
            Operator.LESSEQUAL: operator.le,
            Operator.EQUAL: operator.eq,
            Operator.SIMILAR: operator.eq,
            Operator.GREATEREQUAL: operator.ge,
            Operator.GREATER: operator.gt,
            Operator.UNEQUAL: operator.ne,
        }
        return comparators[self](first, second)


def parse_version(s, minlength=0, strict=False):
    """
    Resolve version numbers

    Examles given at https://packaging.python.org/en/latest/discussions/versioning/

    >>> parse_version('1.2.0')
    >>> parse_version('1.2.0.dev1')
    '1.2.0.9999.1'
    >>> parse_version('1.2.0a1')
    '1.2.0_alpha1'
    >>> parse_version('1.2.0b1')
    '1.2.0_beta1'
    >>> parse_version('1.2.0rc1')
    '1.2.0_rc1'
    >>> parse_version('1.2.0.post1')
    '1.2.0_p1'
    >>> parse_version('1.2.0a1.post1')
    '1.2.0_alpha1_p1'
    >>> parse_version('23.12')
    '23.12'
    >>> parse_version('42')
    '42'
    >>> parse_version('1!1.0')
    'uff'
    # Local version modifiers
    >>> parse_version('0.5.dev1+gd00980f')
    >>> parse_version('0.5.dev1+gd00980f.d20231217')
    """
    if mo := re.fullmatch(r'(rev|v)?([0-9]+[\.0-9]*)(.*)', s.strip(), re.I):
        _, version, tail = mo.groups('0')
        components = tuple(map(int, filter(None, version.split('.'))))
        if len(components) < minlength:
            components += (0,) * (minlength - len(components))
        parts = {'components': components}
        if tail:
            if mo := re.fullmatch(r'[-_\.]?(a|alpha)[-_\.]?([0-9]+)',
                                  tail, re.I):
                parts['alpha'] = int(mo.group(2))
            elif mo := re.fullmatch(r'[-_\.]?(b|beta)[-_\.]?([0-9]+)',
                                    tail, re.I):
                parts['beta'] = int(mo.group(2))
            elif mo := re.fullmatch(r'[-_\.]?(dev|pre)[-_\.]?r?([0-9]+)',
                                    tail, re.I):
                parts['pre'] = int(mo.group(2))
            elif tail == 'dev':
                parts['pre'] = 0
            elif mo := re.fullmatch(r'[-_\.]?rc[-_\.]?([0-9]+)',
                                    tail, re.I):
                parts['rc'] = int(mo.group(1))
            elif mo := re.fullmatch(r'[-_\.]?(p|post)?[-_\.]?([0-9]+)',
                                    tail, re.I):
                parts['p'] = int(mo.group(2))
            elif tail == '*':
                pass
            else:
                if strict:
                    raise ValueError(f'Invalid version `{s}`.')
                _logger.warn(f'Omitted version tail `{tail}`.')
        return Version(**parts)
    else:
        if strict:
            raise ValueError(f'Unparsable version `{s}`.')
        _logger.warn(f'Unparsable version `{s}`.')
        return Version((0,) * max(2, minlength))


def parse_operator(s):
    match s.strip():
        case '<':
            return Operator.LESS
        case '<=':
            return Operator.LESSEQUAL
        case '==':
            return Operator.EQUAL
        case '===':
            return Operator.EQUAL
        case '~=':
            return Operator.SIMILAR
        case '>=':
            return Operator.GREATEREQUAL
        case '>':
            return Operator.GREATER
        case '!=':
            return Operator.UNEQUAL
        case _:
            _logger.warn(f'Unparsable operator `{s}`.')
            return Operator.GREATEREQUAL


def extract_requires_python(requires_python):
    default_py_versions = list(sorted(PYTHON_VERSIONS))

    if not requires_python or not requires_python.strip():
        return default_py_versions

    # clean real world data
    requires_python = requires_python.replace(' ', '')

    req_atoms = list(map(lambda s: s.strip(),
                         requires_python.split(',')))
    req_parsed = []
    for req_atom in req_atoms:
        if mo := re.fullmatch(r'([=<>!~]+)(.+)', req_atom):
            op = parse_operator(mo.groups()[0])
            version = parse_version(mo.groups()[1], minlength=2)
            req_parsed.append((op, version))
        else:
            _logger.warn(f'Unhandled requires_python atom `{req_atom}`!')

    py_versions = list(sorted(PYTHON_VERSIONS))
    for op, version in req_parsed:
        py_versions = [v for v in py_versions if op.compare(v, version)]
    if (not py_versions
        and any(op in {Operator.EQUAL, Operator.SIMILAR}
                for op, _ in req_parsed)):
        # Fix for broken version specs in the wild.
        # Some packages supporting e.g. 3.7 and above wrongly depend on ~=3.7.
        _logger.warn(f'Used default py for boguous spec `{requires_python}`.')
        return default_py_versions
    return py_versions


def requires_python_from_classifiers(classifiers):
    default_py_versions = list(sorted(PYTHON_VERSIONS))
    classifiers = set(classifiers)

    ret = []
    for version in default_py_versions:
        if f"Programming Language :: Python :: {version}" in classifiers:
            ret.append(version)
    return ret


def extract_requires_dist(requires_dist, substitutions):
    ret = []
    if not requires_dist:
        return ret
    for entry in requires_dist:
        if mo := re.fullmatch(
                (r'([-_\.a-zA-Z0-9]+)\s*(\[[-_a-zA-Z0-9,\s]+\])?\s*'
                 r'([(=<>!~][=<>!~0-9a-zA-Z\.(),\s\*]+)?\s*(;.*)?'),
                entry.strip()):
            name, _, versionbounds, conditions = mo.groups()
            # We ignore the extra in the dependency spec to avoid collisions
            # (or better lack of hits) with the main tree

            dep = {
                'name': sanitize_package_name(resolve_package_name(
                    name, substitutions)),
                'versionbound': None,
                'extras': [],
            }

            if versionbounds and versionbounds.strip():
                opranking = [
                    Operator.EQUAL,
                    Operator.SIMILAR,
                    Operator.LESS,
                    Operator.LESSEQUAL,
                    Operator.GREATER,
                    Operator.GREATEREQUAL,
                    Operator.UNEQUAL,
                ]
                cleaned = ''.join(c for c in versionbounds
                                  if c not in ' ()')
                topop, topversion = None, None
                for part in cleaned.split(','):
                    if mobj := re.fullmatch(r'([=<>!~]+)(.+)', part.strip()):
                        # FIXME in the case `==2.6.*` we may want to relax the
                        # resulting dependency to `<=2.7` or `>=2.6` instead
                        # of using `==2.6.0`
                        op = parse_operator(mobj.groups()[0])
                        version = parse_version(mobj.groups()[1], minlength=2)
                        if topop is None or (opranking.index(op)
                                             < opranking.index(topop)):
                            topop, topversion = op, version
                    else:
                        _logger.warn(f'Unhandled version bound `{part}`.')
                if topop:
                    opencoding = {
                        Operator.EQUAL: '~',
                        Operator.SIMILAR: '~',
                        Operator.LESS: '<',
                        Operator.LESSEQUAL: '<=',
                        Operator.GREATER: '>',
                        Operator.GREATEREQUAL: '>=',
                        Operator.UNEQUAL: '>',
                    }
                    dep['versionbound'] = (opencoding[topop], topversion)

            skip = False
            if conditions:
                cleaned = ''.join(c for c in conditions if c not in '();')
                terms = [term.strip()
                         for clause in cleaned.split(' and ')
                         for term in clause.split(' or ')]
                for term in terms:
                    term = term.replace(' ', '')
                    if mobj := re.fullmatch(
                            r'''extra\s*==\s*['"]([-_\.a-zA-Z0-9]+)['"]''',
                            term):
                        dep['extras'].append(
                            sanitize_useflag(mobj.groups()[0]))
                    elif (term.startswith('platform_python_implementation')
                          or term.startswith('implementation_name')):
                        # FIXME handle python implementation differences
                        pass
                    elif (term.startswith('python_version')
                          or term.startswith('python_full_version')):
                        op = parse_operator(''.join(
                            c for c in term if c in '=<>!~'))
                        version = parse_version(''.join(
                            c for c in term if c in '0123456789.'))
                        if not any(op.compare(available, version)
                                   for available in PYTHON_VERSIONS):
                            skip = True
                        # FIXME if only some versions match it would be nice
                        # to make this into a conditional dependency like so:
                        # $(python_gen_cond_dep 'dev-python/tomli' 3.{9..10})
                    elif ((term.startswith('os_name')
                           or term.startswith('platform_system')
                           or term.startswith('sys_platform'))
                          and (('windows' in term.lower()
                                or 'nt' in term.lower()
                                or 'win32' in term.lower())
                               and '==' in term)):
                        skip = True
                    elif (term.startswith('os_name')
                          or term.startswith('platform_system')
                          or term.startswith('sys_platform')):
                        # FIXME handle platform differences
                        pass
                    elif term.startswith('platform_machine'):
                        # FIXME handle architecture differences
                        pass
                    else:
                        # FIXME handle more
                        _logger.warn(f'Ignoring dependency'
                                     f' condition `{term}`.')
            if not skip:
                ret.append(dep)
        else:
            _logger.warn(f'Dropping unexpected dependency `{entry}`.')
    return ret


def resolve_package_name(package, substitutions):
    normalized = pypi_normalize(package)
    return substitutions.get(normalized, normalized)


def sanitize_package_name(package):
    ret = DBGenerator.filter_characters(package.replace('.', '-'), [
            ('a', 'z'), ('A', 'Z'), ('0', '9'), '+_-'])
    if '-' in ret:
        # Fixup invalid package name due to suffix that looks like a version.
        # Note that captial letters seem to be allow by PMS but are forbidden
        # by pkgcore, so we play safe.
        parts = ret.split('-')
        if len(parts) > 1 and re.fullmatch(r'([0-9\.]+)[a-zA-Z]?', parts[-1]):
            ret = '-'.join(parts[:-1]) + '_' + parts[-1]
    # guarantee that the package name starts with a letter or number
    if not re.match(r'^[a-zA-Z0-9]', ret):
        ret = 'x' + ret
    return ret


def sanitize_useflag(useflag):
    ret = DBGenerator.filter_characters(useflag.replace('.', '-'), [
            ('a', 'z'), ('A', 'Z'), ('0', '9'), '+_-@'])
    # guarantee that the useflag starts with a letter or number
    if not re.match(r'^[a-zA-Z0-9]', ret):
        ret = 'x' + ret
    return ret

class PyPIjsonDataPackageDB(PackageDB):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    def __iter__(self):
        _logger.info("PyPIjsonDataPackageDB")
        return PyPIjsonDataIterator(PyPIjsonDataFromZip(self.persistent_datadir / "main.zip"))

class PyPIjsonDataIterator(object):
    """
    Iterator producing a package_db.database.items() structure (see Iterator class in g-sorcery package_db.py)
    """

#    def __init__(self, directory,
#                 persistent_datadir = None,
#                 preferred_layout_version=1,
#                 preferred_db_version=1,
#                 preferred_category_format=JSON_FILE_SUFFIX):
    def __init__(self,reader):
        _logger.info("PyPIjsonDataIterator")
        self.reader = reader

    def __iter__(self):
        _logger.info("PyPIjsonDataIterator iter")
        return ("dev-python",PyPIjsonDataIteratorPackage(self.reader))

class PyPIjsonDataIteratorPackage(object):

    # FIXME How does the repository object come here?
    def __init__(self,db):
        self.reader = db
        self.repository = None

    def __next__(self):
        nxt = None
        # When we don't have a repository, we don't have main.zip open
        if self.repository is None:
            self.repository = self.reader.open_repository()
            self.first_letter_iter = None
            self.second_letter_iter = None
        # When first_letter_iter is None, we aren't iterating yet
        if first_letter_iter is None:
            # FIXME Does this work?
            self.first_letter_iter = self.reader.get_root_dir(self.repository).iterdir()
            self.second_letter_iter = None
        # first_letter_iter is now not None. If we don't have
        # second_letter_iter we may have to dive into secondary directories
        if second_letter_iter is None:
            try:
                nxt = next(first_letter_iter)
                if nxt.is_dir():
                    self.second_letter_iter = self.firstletterdir.iterdir()
                    nxt = None
            except StopIteration:
                # If we're here, then there is no more files in the root
                # (first-level) directory
                self.first_letter_iter = None
                raise
        # If we're here, nxt is not None, second_letter_iter is None and
        # first_letter_iter is not None, then we're dealing with some JSON
        # files in the first-level directory that the original code also tried
        # to capture.
        if nxt is None:
            try:
                nxt = next(self.second_letter_iter)
            except StopIteration:
                # If we're here, then we reached the end of the current second
                # level directory and have to continue reading the next
                # firstlevel directory.
                self.second_letter_iter = None
                pass
        if nxt is not None and nxt.is_file() and nxt.suffix == '.json':
            with nxt.open() as f:
                # Determine package name
                resolved = self.repository.resolve_pn(datapath)
                # Return {package_name: versions[]}
                return {resolved: PyPIjsonDataIteratorVersion(resolved,f)}
        # Recursion either when the second-level directory reached the end or
        # nxt isn't a JSON file we're looking for. This shouldn't loop forever,
        # because the StopIteration of the first-level iterable is raised.
        return __next__(self)

class PyPIjsonDataIteratorVersion(object):
    """
    Iterator effectively parsing through one PyPI JSON file

    >>> import gs_pypi.pypi_db,pprint
    >>> i=gs_pypi.pypi_db.PyPIjsonDataIteratorVersion("transform3d",open("tests/transform3d.json"))
     * Processing transform3d.
     * Creating 0.0.0.
     * Processing transform3d.
     * Creating 0.0.1.
     * Processing transform3d.
     * Creating 0.0.2.
     * Processing transform3d.
     * Creating 0.0.3.
     * Processing transform3d.
     * Creating 0.0.4.
    """

    def __init__(self,name,f):
        self.iter = {}
        for ver,data in json.load(f).items():
            self.iter[ver] = PyPIjsonDataIteratorEbuildData(name,ver,data)

    def __iter__(self):
        return iter(self.iter.items())

class PyPIjsonDataIteratorEbuildData(object):
    """
    PackageDB iterator that creates ebuild data dict's on the fly

    This class needs to provide the exact interfaces PackageDB needs. The
    doctest below tests this.

    >>> import gs_pypi.pypi_db,pprint
    >>> i=gs_pypi.pypi_db.PyPIjsonDataIteratorVersion("transform3d",open("tests/transform3d.json"))
	 * Processing transform3d.
	 * Creating 0.0.0.
	 * Processing transform3d.
	 * Creating 0.0.1.
	 * Processing transform3d.
	 * Creating 0.0.2.
	 * Processing transform3d.
	 * Creating 0.0.3.
	 * Processing transform3d.
	 * Creating 0.0.4.
    >>> for ver,ebuild_data in iter(i):
    ...     print (ver)
    ...     ebuild_data=dict(ebuild_data)
    ...     pprint.pprint(ebuild_data)
    ...     break
    ... 
    0.0.0
    {'dependencies': <g_sorcery.g_collections.serializable_elist object at 0x7f02da3edbd0>,
     'description': 'handy classes for 3d transformations',
     'distutils_use_pep517': 'standalone',
     'homepage': 'https://github.com/RasmusHaugaard/transform3d',
     'iuse': '',
     'license': '',
     'literalname': '${PN}',
     'mtime': '2020-07-31T15:33:18.437661+00:00',
     'python_compat': '( python{3_10,3_11,3_12} )',
     'realname': '${PN}',
     'realversion': '${PV}',
     'repo_uri': 'https://files.pythonhosted.org/packages/source/${REALNAME::1}/${REALNAME}/',
     'sourcefile': '${REALNAME}-${REALVERSION}.tar.gz',
     'src_uri': <gs_pypi.pypi_db.SourceURI object at 0x7f02da3ffe10>}
    """

    def __init__(self,package,version,data):
        self.json_data = data
        self.ebuild_data = None
        self.pipeline = PyPIpeline()
        self.pipeline.set_pkg_db(self)
        self.pipeline.process_version(package, {version: data})
    def in_category(self, category, name):
        return False

    def add_package(self,package,ebuild_data):
        """
        Compatible with pkg_db interface. Receives the database call to add a package
        """
        self.package = package
        self.ebuild_data = ebuild_data

    def set_common_data(self, category, common_data):
        pass

    def add_category(self,category):
        """
        We already know it's gonna be dev-python for obvious reasons.
        """
        pass

    def __iter__(self):
        # We return one ebuild_data JSON. 
        # PyPIpeline generates it during self.__init__.
        return iter(self.ebuild_data.items())

    def __getitem__(self,key):
        """
        Support accessing data in ebuild_data directly
        from https://stackoverflow.com/questions/39286116/python-class-behaves-like-dictionary-or-list-data-like
        """
        return self.ebuild_data[key]

    def __setitem__(self,key,value):
        self.ebuild_data[key]=value

class PypiDBGenerator(DBGenerator):
    """
    Implementation of database generator for PYPI backend.
    """

    def __call__(self,*args,**kwargs):
        """
        Create package database and place our dynamic hooks in it
        """
        _logger.info("Opening ")
        pkg_db = super().__call__(*args,**kwargs)
        z=PyPIjsonDataFromZip(pkg_db.persistent_datadir + "/main.zip")
        # Hack dynamic package info generation into g-sorcery pkg database
        pkg_db.database = PyPIjsonDataIterator(z)
        return(pkg_db)

    def generate_tree(self, pkg_db, common_config, config):
        # The g-sorcery interface asks me to carry along pkg_db and config's
        # C-style. However, we're in python.
        self.pkg_db = pkg_db
        self.common_config = common_config
        self.config = config
        #
        self.exclude = set(self.combine_config_lists(
            [common_config, config], 'exclude'))
        self.wanted = set(self.combine_config_lists(
            [common_config, config], 'wanted'))
        self.substitutions = self.combine_config_dicts(
            [common_config, config], 'substitute')
        self.nonice = set(self.combine_config_lists(
            [common_config, config], 'nonice'))
        self.mainpkgs = self._lookupmaintree()
        # Now proceed with normal flow
        super().generate_tree(pkg_db, common_config, config)

    def process_data(self, pkg_db, data, common_config, config):
        reader = PyPIjsonDataFromZip(self.pkg_db.persistent_datadir + "/main.zip")
        pipeline = PyPIjsonDataToPipeline(reader,pkg_db)

    def _lookupmaintree(self):
        """
        Look up main tree by accessing Gentoo git repository via web

        FIXME: We can safely assume that g-sorcery mostly will be run on Gentoo
        machines, so we should instead take a look into the local Gentoo
        repository in /var/db/repos/gentoo

        """
        ret = set()
        fname = "dev-python.html"
        pattern = (
            r'<a[^>]*/gentoo.git/tree/dev-python[^>]*>([-a-zA-Z0-9\._]+)</a')
        with tempfile.TemporaryDirectory() as download_dir:
            if wget(self.config['gentoo_main_uri'], download_dir, fname):
                raise DownloadingError("Retrieving main tree directory failed")
            with open(pathlib.Path(download_dir) / fname) as htmlfile:
                for line in htmlfile.readlines():
                    if 'd---------' in line:
                        if mo := re.search(pattern, line):
                            ret.add(mo.group(1))
        _logger.info(f'Total of main tree packages: {len(ret)}.')
        return ret

    def get_download_uries(self, common_config, config):
        """
        Get URI of packages index.
        """
        _logger.info('Retrieving package index.')
        return [{"uri": config["data_uri"], "open_file": False, "parser":self.handle_download}]

    def handle_download(self,download):
        """
        Handle downloaded main.zip from github pypi-json-data
        """
        _logger.info('Copying package index to %s' % self.pkg_db.persistent_datadir + "/main.zip")
        if not os.path.exists(self.pkg_db.persistent_datadir):
            os.mkdir(self.pkg_db.persistent_datadir)
        shutil.copyfile(download,self.pkg_db.persistent_datadir + "/main.zip")

class PyPIjsonDataRepository(object):

    """
    The interface needed by the underlying
    filesystem access of this call is
    importlib.resources.abc.Traversable
    """

    def resolve_pn(self,datapath):
        package = datapath.stem
        if package in self.exclude:
            return None
        if (not os.environ.get('GSPYPI_INCLUDE_UNCOMMON')
                and package not in self.wanted):
            # we only include a selected set of packages as otherwise the
            # overlay becomes unwieldy
            return None
        if package != pypi_normalize(package):
            _logger.warn(f'Unnormalized input package {package}.')
        resolved = resolve_package_name(package, self.substitutions)
        return resolved

    def _adjust_datapath(self,datapath,package,resolved):
        if resolved != package:
            alternative = datapath.parent / f'{resolved}.json'
            if alternative.exists():
                _PyPIjsonDataToPipelinelogger.info(f'Switching data source {package} to {resolved}.')
                datapath = alternative
        return datapath

    def parse_datum(self, datapath, datafile):
        resolved = self.resolve_pn(datapath)
        if resolved is None:
            return {} 
        # FIXME I don't get why we're changing the datapath when it's not used
        # afterwards
        datapath = self._adjust_datapath(datapath,datapath.stem,resolved)
        data = json.load(datafile)
        return {resolved: data}

    def set_processor(self,proc):
        self.processor = proc

    def open_repository(self):
        """
        Opens the repository
        """
        raise Exception("Not implemeneted")

    def get_root_dir(self,z):
        """
        Returns a path object representing the root of the repository
        """
        raise Exception("Not implemeneted")

    def close_repository(self):
        """
        Close repository and return any value
        """
        pass

    def process_file(self,entry,f):
        self.processor.process_file(entry,f)

    def parse_data(self):
        """
        Parse package data.
        """
        with self.open_repository() as z:
            firstletterdirs = self.get_root_dir(z)
            for firstletterdir in firstletterdirs.iterdir():
                # There exist some metadata files which do not interest us
                if firstletterdir.is_dir():
                    for second in firstletterdir.iterdir():
                        if second.is_dir():
                            for entry in second.iterdir():
                                if entry.is_file() and entry.suffix == '.json':
                                    with entry.open() as f:
                                        self.process_file(entry,f)
                        elif second.is_file() and second.suffix == '.json':
                            # Some entries are on the first level
                            with second.open() as f:
                                self.process_file(second,f)
        return self.close_repository()

class PyPIjsonDataFromZip(PyPIjsonDataRepository):
    """
    Read ZIP file with PyPI package data in JSON format
    >>> import gs_pypi.pypi_db
    >>> z=gs_pypi.pypi_db.PyPIjsonDataFromZip("tests/main.zip")
    >>> z.set_processor(gs_pypi.pypi_db.PyPIdump())
    >>> try:
    ...     z.parse_data()
    ... except gs_pypi.pypi_db.DoctestFinishedException:
    ...     pass
    ...
     * 0-._.-._.-._.-._.-._.-._.-0.json
     * 0-core-client.json
     * 0-orchestrator.json
     * 0.0.1.json
     * 0.618.json
     * 0.json
     * 00-viet-nam-on-top-00.json
     * 000.json
     * 00000.json
     * 0000000.json
     * 00000000.json
    >>>
    """

    def __init__(self,mainzip):
        self.mainzip = mainzip

    def open_repository(self):
        """
        Reset package data and open ZIP file
        """
        self.zip = zipfile.ZipFile(self.mainzip)
        return self.zip

    def get_root_dir(self,z):
        """
        Get the root directory to traverse

        FIXME Why pass z?
        """
        #_logger.info(z)
        return zipfile.Path(z, at="pypi-json-data-main/release_data/")

    def close_repository(self):
        self.zip.close()

class DoctestFinishedException(Exception):
    """
    doctest helper exception, raised when 10 files have been dumped
    """
    pass

class PyPIdump(object):
    """
    Helper class for doctests: Dumps the first 10 entries of main.zip
    """
    def __init__(self):
        self.remaining = 10

    def process_file(self,entry,f):
        _logger.info( entry.name )
        self.remaining = self.remaining-1
        if self.remaining<0:
            raise DoctestFinishedException("Done")

class SourceURI(object):
    """
    Capture SRC_URI together with checksums

    >>> src_uri=gs_pypi.pypi_db.SourceURI(uri="https://files.pythonhosted.org/packages/16/fc/764c31a0ced73481d033d7a267d185f865abeeb073c410b2fdff9680504f/transform3d-0.0.0-py3-none-any.whl")
    >>> print(src_uri)
    https://files.pythonhosted.org/packages/16/fc/764c31a0ced73481d033d7a267d185f865abeeb073c410b2fdff9680504f/transform3d-0.0.0-py3-none-any.whl
    >>> print(src_uri.split("/"))
    ['https:', '', 'files.pythonhosted.org', 'packages', '16', 'fc', '764c31a0ced73481d033d7a267d185f865abeeb073c410b2fdff9680504f', 'transform3d-0.0.0-py3-none-any.whl']
    >>> src_uri.setHash("md5","4d64f08f869dc9c8b3f8affa03b37e5f")
    >>> src_uri.hashes
    {'MD5': '4d64f08f869dc9c8b3f8affa03b37e5f'}
    >>> print(src_uri)
    https://files.pythonhosted.org/packages/16/fc/764c31a0ced73481d033d7a267d185f865abeeb073c410b2fdff9680504f/transform3d-0.0.0-py3-none-any.whl
    
    """

    # How hash functions in PyPI database match to hash functions in portage Manifest
    hash_translation_table = {
        "md5": "MD5",
        "sha256": "SHA256"
        }

    def __init__(self,uri=None,hashes=None):
        self.uri = uri
        self.hashes = {}
        if hashes is not None:
            self.hashes_from_pypi(hashes)

    def hashes_from_pypi(self,hashes):
        # FIXME Use the map() function
        for h,v in hashes.items():
            self.setHash(h,v)

    def setHash(self,h,v):
        """
        Set single hash
        """
        self.hashes[self.hash_translation_table[h]] = v

    def __getattr__(self,name):
        """
        Route all accesses of data within the class to the string object. 
        This preserves compatibility with old g-sorcery.
        """
        return getattr(self.uri,name)

    def __str__(self):
        """
        Return this object as a string. This preserves compatibility with old g-sorcery.
        """
        return self.uri

class PyPIjsonDataToPipeline(object):
    """
    Processor for PyPIjsonDataRepository type classes who uses PyPIpeline for the processing

    Th
    """

    def __init__(self,sub,db = None):
        """
        Constructor

        sub:PyPIjsonDataRepository
        db:PackageDB
        """
        self.sub = sub
        self.sub.set_processor(self)
        self.pipeline = PyPIpeline()
        if db is not None:
            self.pipeline.set_pkg_db(db)
        self.pipeline.process_init()

    def set_pkg_db(self,db):
        self.pipeline.set_pkg_db(db)

    def process_file(self,entry,f):
        d = self.sub.parse_datum(entry,f)
        for version_json in self.pipeline.process_versions(d):
            self.pipeline.process_version(version_json)

class PyPIpeline(object):
    """
    Central pipeline the convert PyPI structure into portage structure

    This class implements the core of what's necessary to take the contents of
    a PyPI JSON file as input one version at a time and create an ebuild_data
    dict as it can be found in the g-sorcery package database as output.

    We need a pull based method (fetching data one record at a time) due to the
    size of the data we want to convert if it should be possible to create a
    repository containing ALL PyPI packages. This class has also the leftovers
    from the push method which generates ebuild_data for inclusion into the
    PackageDB upon sync. Doing this gets us OOM even on a 64GB RAM machine.

    Base functionality of PyPIpeline is implicitely tested with class
    PyPIjsonDataIteratorEbuildData doctest which uses it.
    """
    
    @staticmethod
    def name_output(package, filtered_package):
        ret = package
        if package != filtered_package:
            ret += f" (as {filtered_package})"
        return ret

    def set_pkg_db(self,pkg_db):
        self.pkg_db = pkg_db

    def _is_package_addable(self, package, data):
        """
        Check if the version of the package is already present in the package database and add it if it is not

        This filters out packages already present in portage as well as packages
        that for some reason have previously already been added.

        Input:
        * package - Package instance
        * data - dict representing the data that has to be written to the ebuild

        Output:
        * call to pkg_db.add_package if the package is not already present
        """
        nout = PyPIpeline.name_output(data['realname'], package.name)
        if self.pkg_db.in_category(package.category, package.name):
            versions = self.pkg_db.list_package_versions(package.category,
                                                    package.name)
            if package.version in versions:
                _logger.warn(f"Rejected package {nout} for collision.")
                return False
        return True

    def process_init(self):
        """
        Initialise common data
        """
        category = "dev-python"
        self.pkg_db.add_category(category)

        common_data = {}
        common_data["eclasses"] = ['g-sorcery', 'gs-pypi']
        common_data["maintainer"] = [{'email': 'gentoo@houseofsuns.org',
                                      'name': 'Markus Walter'}]
        self.pkg_db.set_common_data(category, common_data)


    def process_data(self, data):
        """
        Process parsed package data from PyPI and call process_datum for each package

        Input:
        * data - raw PyPI database parsed into a JSON dict 

        Output:
        * pkg_data given to self.process_datum containing the pypi-json-data
          for one single package (for one single software project)
        """
        self.process_init()

        for package, pkg_data in data['main.zip'].items():
            self.process_datum(package, pkg_data)

    def process_versions(self, package, pkg_data):
        """
        Processes one file of pypi-json-data

        Input:
          package: What package this is
          pkg_data: pypi-json-data JSON dict for only the package above
        Output:
          call to self.process_version with the dict for that version

        """

        versions=[]

        for version in pkg_data.keys():
            _logger.info( "Looking at verion %s of package %s" % (version,package) )
            v=self.process_version(package, {version: pkg_data[version]})
            if v is not None:
                versions.append(v)
        return versions

    #@containment
    def process_version(self, package, pkg_data):
        """
        Go through all variants of one parsed package datum and select the variant we want to utilise in the ebuild
        If you replace the call to process_versions to a call to
        process_version, it will pick one version amongst all available
        versions. For this to work process_version picks each version, but
        passes them in pkg_data as an array containing a single version only.

        """
        _logger.info(f'Processing {package}.')

        fromiso = datetime.datetime.fromisoformat

        def is_prod(v):
            return all(part is None for part in [v.alpha, v.beta, v.pre, v.rc])

        select = {
            variant: {
                'key': None,
                'pkg_datum': None,
                'src_uri': None,
                'use_wheel': 'wheel' in variant,
                'aberrations': [],
            }
            for variant in ['top', 'top-wheel', 'prod', 'prod-wheel',
                            'new', 'new-wheel']
        }
        select['top']['extract'] = select['top-wheel']['extract'] = (
            lambda d: parse_version(d['info']['version']))
        select['prod']['extract'] = select['prod-wheel']['extract'] = (
            lambda d: (is_prod(v := parse_version(d['info']['version'])), v))
        select['new']['extract'] = select['new-wheel']['extract'] = (
            lambda d: min(fromiso(entry['upload_time_iso_8601'])
                          for entry in d['urls']))

        def score_wheel(filename):
            ret = 0
            if mo := re.fullmatch(r'.*-([^-]+)-([^-]+)-([^-]+)\.whl',
                                  filename, re.I):
                python, abi, platform = mo.groups()
                python_scores = {
                    'py3': 200,
                    r'py2\.py3': 200,
                    'cp311': 102,
                    'cp310': 101,
                    'cp3': 100,
                }
                for pattern, bounty in python_scores.items():
                    if re.match(pattern, python):
                        ret += bounty
                        break
                abi_scores = {
                    'none': 300,
                    'py3': 200,
                    r'py2\.py3': 200,
                    'cp312': 103,
                    'cp311': 102,
                    'cp310': 101,
                    'cp3': 100,
                }
                for pattern, bounty in abi_scores.items():
                    if re.match(pattern, abi):
                        ret += bounty
                        break
                platform_scores = {
                    'any': 300,
                    'linux_(x86_64|amd64)': 200,
                    'manylinux.*(x86_64|amd64)': 100,
                }
                for pattern, bounty in platform_scores.items():
                    if re.match(pattern, platform):
                        ret += bounty
                        break
            else:
                _logger.warn(f'Improper wheel file name {filename}')
            return ret

        for datum in reversed(pkg_data.values()):
            for variant in select.values():
                if not datum['urls']:
                    continue
                key = variant['extract'](datum)
                if variant['key'] is not None and variant['key'] >= key:
                    continue
                wheel_score = 0
                for entry in datum['urls']:
                    if variant['use_wheel']:
                        if entry['packagetype'] == 'bdist_wheel':
                            score = score_wheel(entry['filename'])
                            if score > wheel_score:
                                wheel_score = score
                                variant.update({
                                    'key': key,
                                    'pkg_datum': datum,
                                    'src_uri': SourceURI(uri=entry['url'],hashes=entry["digests"]),
                                })
                    else:
                        if entry['packagetype'] == 'sdist':
                            variant.update({
                                'key': key,
                                'pkg_datum': datum,
                                'src_uri': SourceURI(uri=entry['url'],hashes=entry["digests"]),
                            })
                            break

        def ref(variant):
            if variant not in select:
                return None
            return select[variant]['pkg_datum']['info']['version']

        for variant in list(select):
            if select[variant]['key'] is None:
                del select[variant]
        for variant in ['top', 'prod', 'new']:
            if variant not in select or f'{variant}-wheel' not in select:
                continue
            if select[variant]['key'] < select[f'{variant}-wheel']['key']:
                select[variant]['aberrations'].append(
                    f"{variant}-max {select[f'{variant}-wheel']['key']}")
            else:
                del select[f'{variant}-wheel']
        for suffix in ['', '-wheel']:
            if ref(f'top{suffix}') == ref(f'prod{suffix}') is not None:
                del select[f'prod{suffix}']
            if ref(f'top{suffix}') == ref(f'new{suffix}') is not None:
                del select[f'new{suffix}']
            if ref(f'prod{suffix}') == ref(f'new{suffix}') is not None:
                del select[f'new{suffix}']
        if len(allref := list(map(ref, select))) > len(set(allref)):
            _logger.warn(f'Redundant variants selected: {allref}'
                         f' by {list(select)}')

        if not select:
            _logger.warn(f'No valid releases for {package} -- dropping.')

        # Create a package for the selected variants
        variants=[]
        for variant in select.values():
            variants.append(variant)
            self.create_package(
                package, variant['pkg_datum'],
                variant['src_uri'], variant['use_wheel'],
                variant['aberrations'])
        return variants

    def get_ebuild_data(self, package, pkg_datum,
                       src_uri, use_wheel, aberrations):
        """
        Assemble all the data needed to create a package ebuild file

        Input: 
        * pkg_db & config
        * package:
        * pkg_datum: as it can be found pypi-json-data

        Output passed to self.maybe_add_package:
        * pkg_db
        * Package instance - basically an object representing the contents of ${P}
        * ebuild_data: dict with all the fields needed to construct an ebuild
        """
        _logger.info(f'Creating {pkg_datum["info"]["version"]}.')
        category = "dev-python"
        homepage = pkg_datum['info']['home_page'] or ""
        if not homepage:
            purls = pkg_datum['info'].get('project_urls') or {}
            for key in ["Homepage", "homepage"]:
                homepage = purls.get(key, "")
                if homepage:
                    break
        homepage = DBGenerator.escape_bash_string(DBGenerator.strip_characters(homepage))

        fromiso = datetime.datetime.fromisoformat
        mtime = min(fromiso(entry['upload_time_iso_8601'])
                    for entry in pkg_datum['urls'])

        pkg_license = pkg_datum['info']['license'] or ''
        # This has to avoid any characters that have a special meaning for
        # dependency specification, these are: !?|^()
        pkg_license = DBGenerator.filter_characters(
            (pkg_license.splitlines() or [''])[0],
            mask_spec=[
                ('a', 'z'), ('A', 'Z'), ('0', '9'),
                ''' #%'*+,-./:;=<>&@[]_{}~'''])
        # FIXME config is relevant here
        #pkg_license = DBGenerator.convert([], "licenses",
        #                           pkg_license)
        pkg_license = DBGenerator.escape_bash_string(pkg_license)

        requires_python = extract_requires_python(
            pkg_datum['info']['requires_python'])
        for addon in requires_python_from_classifiers(
                pkg_datum['info'].get('classifiers', [])):
            if addon not in requires_python:
                requires_python.append(addon)
        if not requires_python:
            _logger.warn(f'No valid python versions for {package}'
                         f' -- dropping.')
            return
        py_versions = list(map(
            lambda version: f'{version.components[0]}_{version.components[1]}',
            requires_python))

        if len(py_versions) == 1:
            python_compat = '( python' + py_versions[0] + ' )'
        else:
            python_compat = '( python{' + (','.join(py_versions)) + '} )'

        # FIXME Fix this
        self.substitutions = {}
        requires_dist = extract_requires_dist(
            pkg_datum['info']['requires_dist'], self.substitutions)

        dependencies = []
        useflags = set()
        # FIXME
        self.mainpkgs = {}
        for dep in requires_dist:
            for extra in (dep['extras'] or [""]):
                if (dep['name'] in self.mainpkgs) and dep["versionbound"]:
                    # keep version bounds for packages in the main tree as
                    # there will probably be some choice in the relevant cases
                    dop, dver = dep["versionbound"]
                else:
                    # ignore version bound as we only provide the most recent
                    # version anyway so there is no choice. Additionally this
                    # fixes broken dependency specs where there either is an
                    # error or which are simply outdated.
                    dop, dver = "", ""
                dependencies.append(Dependency(
                    category, dep['name'], usedep='${PYTHON_USEDEP}',
                    useflag=extra, version=str(dver), operator=str(dop)))
                if extra:
                    useflags.add(extra)
        if use_wheel:
            dependencies.append(Dependency("virtual", "allow-pypi-wheels"))

        filtered_package = sanitize_package_name(package)
        # for accounting note the actual name of the package
        # MY_PN=
        literal_package = pkg_datum['info']['name']
        # MY_PV=
        version = pkg_datum["info"]["version"]
        try:
            filtered_version = str(parse_version(version, strict=True))
        except ValueError:
            bad_version = version
            filtered_version = "%04d%02d%02d" % (
                mtime.year, mtime.month, mtime.day)
            _logger.warn(f'Version {bad_version} is bad'
                         f' using {filtered_version}.')
            aberrations.append(f"badver {bad_version}")

        nice_src_uri = src_uri
        filename = src_uri.split('/')[-1]
        pattern = (r'https://files\.pythonhosted\.org/packages'
                   r'/[0-9a-f]+/[0-9a-f]+/[0-9a-f]+/.*')
        if re.fullmatch(pattern, src_uri.lower()):
            filepath = src_uri.removesuffix(filename)
            suffix = ''
            for extension in ['.tar.gz', '.tar.bz2', '.zip']:
                if filename.endswith(extension):
                    suffix = extension
                    filename = filename.removesuffix(extension)
                    break
            src_uri_filters = [
                (f'{version}', '${REALVERSION}'),
                (f'{package}', '${REALNAME}'),
                (f'{literal_package}', '${LITERALNAME}'),
                (f'{package.replace("-", "_")}', '${REALNAME//-/_}'),
                (f'{package.replace("_", "-")}', '${REALNAME//_/-}'),
                (f'{literal_package.replace("-", "_")}',
                 '${LITERALNAME//-/_}'),
                (f'{literal_package.replace("_", "-")}',
                 '${LITERALNAME//_/-}'),
            ]
            for pattern, replacement in src_uri_filters:
                filename = filename.replace(pattern, replacement)
            filename = filename + suffix
            npattern = r'\$\{(LITERALNAME|REALNAME)[-_/]*\}-\$\{REALVERSION\}'
            if ((mo := re.match(npattern, filename))
                    and package[0] in string.ascii_letters + string.digits
                    #FIXME
                    and pypi_normalize(package) not in DBGenerator.nonice
                    ):
                name = mo.group(1)
                # Use redirect URL to avoid churn through the embedded hashes
                # in the actual URL
                tag = None
                if use_wheel:
                    if mo := re.fullmatch(r'.*-([^-]+)-[^-]+-[^-]+\.whl',
                                          filename, re.I):
                        tag = mo.group(1)
                else:
                    tag = 'source'
                if tag:
                    nice_src_uri = (
                        f'https://files.pythonhosted.org/packages/{tag}'
                        f'/${{{name}::1}}/${{{name}}}/{filename}')
                else:
                    _logger.warn(f'Unmatched SRC_URI `{src_uri}`.')
                    nice_src_uri = filepath + filename
            else:
                _logger.warn(f'Unsubstituted SRC_URI `{src_uri}`.')
                nice_src_uri = filepath + filename
        else:
            _logger.warn(f'Unexpected SRC_URI `{src_uri}`.')

        description = pkg_datum['info']['summary'] or ''
        if use_wheel:
            aberrations.append('wheel')
        if aberrations:
            description += " [" + ", ".join(aberrations) + "]"
        filtered_description = DBGenerator.escape_bash_string(DBGenerator.strip_characters(
            description))

        ebuild_data = {}
        ebuild_data["realname"] = (
            "${PN}" if package == filtered_package else package)
        ebuild_data["literalname"] = (
            "${PN}" if filtered_package == literal_package
            else literal_package)
        ebuild_data["realversion"] = (
            "${PV}" if version == filtered_version else version)
        ebuild_data["mtime"] = mtime.isoformat()

        ebuild_data["description"] = filtered_description

        ebuild_data["homepage"] = homepage
        ebuild_data["license"] = pkg_license
        ebuild_data["src_uri"] = src_uri
        ebuild_data["sourcefile"] = filename
        ebuild_data["repo_uri"] = nice_src_uri.removesuffix(
            ebuild_data["sourcefile"])
        ebuild_data["python_compat"] = python_compat
        ebuild_data["iuse"] = " ".join(sorted(useflags))
        deplist = serializable_elist(separator="\n\t")
        deplist.extend(dependencies)
        ebuild_data["dependencies"] = deplist
        ebuild_data["distutils_use_pep517"] = (
            "wheel" if use_wheel else "standalone")
        return ebuild_data

    def create_package(self, package, pkg_datum,
                       src_uri, use_wheel, aberrations):
        ebuild_data = self.get_ebuild_data(package, pkg_datum,
                       src_uri, use_wheel, aberrations)
        package = Package("dev-python", ebuild_data['realname'], ebuild_data['realversion'])
        if self._is_package_addable(
                package,
                ebuild_data):
            self.pkg_db.add_package(package, ebuild_data)

    def convert_internal_dependency(self, configs, dependency):
        """
        At the moment we have only internal dependencies, each of them
        is just a package name.
        """
        return Dependency("dev-python", dependency)

# vim:expandtab
