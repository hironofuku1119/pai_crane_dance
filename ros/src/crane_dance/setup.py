from setuptools import setup

package_name = 'crane_dance'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='crane_dance',
    maintainer_email='noreply@example.com',
    description='最終課題の再生ノード（お手本・npz の再生）',
    license='TBD',
    entry_points={
        'console_scripts': [
            'player = crane_dance.player:main',
        ],
    },
)
