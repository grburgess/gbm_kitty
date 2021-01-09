import os
from setuptools import setup
import versioneer


# Create list of data files


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
#    include_package_data=True,
 #   package_data={"": extra_files},
#    scripts=["bin/get_trigdat", "bin/get_posthist"],
)
