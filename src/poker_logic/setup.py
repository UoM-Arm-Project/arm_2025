from glob import glob

from setuptools import find_packages, setup

package_name = 'poker_logic'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kshitij',
    maintainer_email='kshitijjha1008@gmail.com',
    description='Poker decision layer and skill executor for the poker arm.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'game_manager = poker_logic.nodes.game_manager_node:main',
            'skill_executor = poker_logic.nodes.skill_executor_node:main',
            'table_console = poker_logic.nodes.table_console_node:main',
            'poker_sim = poker_logic.sim:main',
        ],
    },
)
