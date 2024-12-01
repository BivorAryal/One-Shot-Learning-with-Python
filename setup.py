from setuptools import find_packages, setup
from typing import List


def get_requirement(file_path:str)->List[str]:
    '''
    This function will return the list of req.
    '''
    requirement = []
    with open(file_path) as file_obj:
        requirement = file_obj.readlines()
        [req.replace("\n","") for req in requirement]

setup(
    name = 'One Shot Learning with Python',
    version= '0.0.1',
    author= 'Bivor Arjayal'
    author_email= 'bivorbivor@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirement.txt')
)