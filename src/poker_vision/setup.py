from glob import glob

from setuptools import find_packages, setup

package_name = 'poker_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'requirements.txt']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kshitij',
    maintainer_email='kshitijjha1008@gmail.com',
    description='YOLO + ArUco vision nodes feeding poker_logic.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'card_detector = poker_vision.nodes.card_detector_node:main',
            'pot_estimator = poker_vision.nodes.pot_estimator_node:main',
            'vision_probe = poker_vision.tools.probe:main',
            'export_model = poker_vision.tools.export_model:main',
            'make_marker = poker_vision.tools.make_marker:main',
        ],
    },
)
