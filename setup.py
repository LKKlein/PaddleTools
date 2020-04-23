from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.install import install

from paddletools import __version__


def load_package_data():
    cur_dir = Path(__file__).resolve().parent
    path = Path(cur_dir.joinpath("paddletools"))
    package_data = {"paddletools": []}

    for data_path in path.glob("**/*"):
        if data_path.is_file():
            package_data["paddletools"].append(str(data_path.relative_to(path)))
    return package_data


class custom_install(install):

    def run(self):
        self.single_version_externally_managed = True
        super(custom_install, self).run()


def load_requirements():
    requirements = []
    with open('./requirements.txt') as requirements_file:
        for line in requirements_file:
            line = line.strip()
            if not line or line.startswith("# "):
                continue
            requirements.append(line)
    return requirements


if __name__ == "__main__":
    setup(
        name="paddletools",
        version=__version__,
        python_requires=">=3.3",
        packages=find_packages(),
        cmdclass={"install": custom_install},
        package_data=load_package_data(),
        install_requires=load_requirements(),
        entry_points={"console_scripts": [
            "pdtools = paddletools.control:main"
        ]},
        keywords=["Paddle", "Tools"],
        url="https://github.com/LKKlein/PaddleTools",
        author="lvkun",
        author_email="lkklein@163.com",
        include_package_data=True,
        license='MIT',
    )
