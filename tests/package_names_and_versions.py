
import unittest
import gs_pypi.pypi_db
import re
import json
import portage.versions

class TestPackageNamingStandardAdherence(unittest.TestCase):
    """
    Test adherence to package name and versioning coherence

    """

    # Code below copied from https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers-regex
    VERSION_PATTERN = r"""
        v?
        (?:
            (?:(?P<epoch>[0-9]+)!)?                           # epoch
            (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
            (?P<pre>                                          # pre-release
                [-_\.]?
                (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
                [-_\.]?
                (?P<pre_n>[0-9]+)?
            )?
            (?P<post>                                         # post release
                (?:-(?P<post_n1>[0-9]+))
                |
                (?:
                    [-_\.]?
                    (?P<post_l>post|rev|r)
                    [-_\.]?
                    (?P<post_n2>[0-9]+)?
                )
            )?
            (?P<dev>                                          # dev release
                [-_\.]?
                (?P<dev_l>dev)
                [-_\.]?
                (?P<dev_n>[0-9]+)?
            )?
        )
        (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
    """

    def setUp(self):
        """
        Set up the tests. 

        We need working access to main.zip in all cases. B-)
        """
        self.zip = gs_pypi.pypi_db.PyPIjsonDataFromZip("tests/main.zip")
        # Code below copied from https://packaging.python.org/en/latest/specifications/name-normalization/
        self.pypi_name_regex = re.compile( 
            r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$",
            re.VERBOSE | re.IGNORECASE,
            )
        # Code below copied from https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers-regex
        self.pypi_version_regex = re.compile(
            r"^\s*" + self.VERSION_PATTERN + r"\s*$",
            re.VERBOSE | re.IGNORECASE,
        )
        # Below we're importing a private variable... lalalalala...
        self.gentoo_name_regex = re.compile(portage.versions._pkg) #re.compile(r"^[A-Za-z0-9-][A-Za-z0-9+_-]*[A-Za-z0-9+_]$")
        self.gentoo_version_regex = portage.versions.ver_regexp #re.compile(r"^[0-9]+(\.[0-9]+)*[a-z]?((_alpha|_beta|_pre|_rc|_p)[0-9]*)*(-r[0-9])?$")

    class CallParent(object):

        def __init__(self,parent):
            self.parent = parent
    
    #
    # Test PyPI names
    #
    class CallPypiName(CallParent):

        def process_file(self,entry,f):
            self.parent.check_pypi_name(entry,f)
    
    def check_pypi_name(self,entry,f):
        """
        Check PyPI package name
        """
        name = gs_pypi.pypi_db.pypi_normalize(entry.stem)
        with self.subTest(name):
            #self.counter-=1
            #if self.counter<0: 
            #    raise TestPackageNamingStandardAdherence.TestFinishedException
            self.assertIsNotNone(self.pypi_name_regex.fullmatch(name))
    
    def test_pypi_names(self):
        self.zip.set_processor(TestPackageNamingStandardAdherence.CallPypiName(self))
        self.counter=10
        try:
            self.zip.parse_data()
        except TestPackageNamingStandardAdherence.TestFinishedException:
            pass
    
    #
    # Test pypi versions
    #
    class CallPypiVersion(CallParent):

        def process_file(self,entry,f):
            self.parent.check_pypi_version(entry,f)

    class TestFinishedException(Exception):
        pass
    
    def check_pypi_version(self,entry,f):
        """
        Interface method called once for each file in main.zip
        """
        # FIXME: Not on the name, on the version
        j=json.load(f)
        for v in j:
            with self.subTest("%s %s" % (entry.name,v) ):
                m=self.pypi_version_regex.fullmatch(v)
                self.assertIsNotNone(m)
            #self.counter-=1
            #if self.counter<0: 
            #    raise TestPackageNamingStandardAdherence.TestFinishedException
    
    def test_pypi_versions(self):
        self.zip.set_processor(TestPackageNamingStandardAdherence.CallPypiVersion(self))
        #self.counter=1000
        try:
            self.zip.parse_data()
        except TestPackageNamingStandardAdherence.TestFinishedException:
            pass
    
    #
    # Test Gentoo names
    #
    def test_gentoo_names(self):
        i = gs_pypi.pypi_db.PyPIjsonDataIteratorPackages(self.zip)
        for package,versions in i.items():
            with self.subTest(package):
                m=self.gentoo_name_regex.fullmatch(package)
                self.assertIsNotNone(m)
    
    #
    # Test Gentoo versions
    #
    def test_gentoo_versions(self):
        i = gs_pypi.pypi_db.PyPIjsonDataIteratorPackages(self.zip)
        #counter=300
        for package,versions in i.items():
            #counter-=1
            #if counter<0: return
            for version in versions:
                with self.subTest("%s %s" % (package,version)):
                    m=self.gentoo_version_regex.fullmatch(version)
                    self.assertIsNotNone(m)
        
if __name__ == '__main__':
    unittest.main()

# vim:expandtab:fileencoding-utf-8
