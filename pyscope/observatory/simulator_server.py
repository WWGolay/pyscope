import multiprocessing
import os
import platform
import signal
import subprocess
from tarfile import TarFile
from zipfile import ZipFile


class SimulatorServer:
    def __init__(self):
        if platform.system() == "Darwin":
            sys_name = "macos-x64"
            zip_type = ".zip"
        elif platform.system() == "Linux":
            sys_name = "linux"
            zip_type = ".tar.xz"

            if "arm" in platform.machine():
                sys_name += "-armhf"
            elif "aarch64" in platform.machine():
                sys_name += "-aarch64"
            elif "x86_64" in platform.machine():
                sys_name += "-x64"

        elif platform.system() == "Windows":
            sys_name = "windows"
            zip_type = ".zip"

            if "x86_64" in platform.machine():
                sys_name += "-x86"
            elif "x64" in platform.machine():
                sys_name += "-x64"

        else:
            raise Exception("Unsupported platform")

        dirname = "ascom.alpaca.simulators." + sys_name

        # p = subprocess.check_output(
        #     f"""gh release download v0.3.1 --repo ASCOMInitiative/ASCOM.Alpaca.Simulators --skip-existing --pattern {dirname}{zip_type}""",
        #     shell=True,
        # )

        # if zip_type == ".zip":
        #     with ZipFile(dirname + zip_type, "r") as zipObj:
        #         zipObj.extractall()
        # elif zip_type == ".tar.xz":
        #     with TarFile.open(dirname + zip_type, "r") as tarObj:
        #         tarObj.extractall()

        # os.chmod(dirname + "/ascom.alpaca.simulators.exe", 0o755)  # added .exe

        # if platform.system() == "Darwin":
        #     self.process = subprocess.Popen(
        #         ("sudo " + dirname + "/ascom.alpaca.simulators"),
        #         preexec_fn=os.setpgrp,
        #         start_new_session=True,
        #         stderr=subprocess.DEVNULL,
        #         stdout=subprocess.DEVNULL,
        #         shell=True,
        #     )
        # elif platform.system() == "Linux":
        #     self.process = subprocess.Popen(
        #         ("sudo " + dirname + "/ascom.alpaca.simulators"),
        #         start_new_session=True,
        #         stderr=subprocess.DEVNULL,
        #         stdout=subprocess.DEVNULL,
        #         shell=True,
        #     )
        # else:
        #     self.process = subprocess.Popen(
        #         (dirname + "/ascom.alpaca.simulators"),
        #         preexec_fn=os.setpgrp,
        #         start_new_session=True,
        #         stderr=subprocess.DEVNULL,
        #         stdout=subprocess.DEVNULL,
        #         shell=True,
        #     )

        self.dirname = dirname + "/ascom.alpaca.simulators.exe"

    def __del__(self):
        subprocess.Popen(f"sudo kill {(os.getpgid(self.process.pid)+1)}", shell=True)
