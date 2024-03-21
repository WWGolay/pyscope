import atexit
import configparser
import errno
import logging
import os
import stat
import threading
import time
from pathlib import Path

import click
import paramiko

logger = logging.getLogger(__name__)

# TODO: Add support for syncing individual files, add support for syncing within local directories


@click.command(
    epilog="""Check out the documentation at
               https://pyscope.readthedocs.io/en/latest/
               for more information."""
)
@click.option(
    "-l",
    "--local-dir",
    "local_dir",
    default="./",
    type=click.Path(exists=True, resolve_path=True),
    help="Local directory to sync",
)
@click.option(
    "-r",
    "--remote-dir",
    "remote_dir",
    default="~/",
    type=str,
    help="Remote directory to sync to",
)
@click.option(
    "-m",
    "--mode",
    default="both",
    type=click.Choice(["send", "receive", "both"]),
    help="Mode to sync in",
)
@click.option(
    "-i",
    "--ignore-dir",
    "ignore_dir",
    multiple=True,
    type=str,
    help="Directories to ignore",
)
@click.option(
    "-e",
    "--ignore-ext",
    "ignore_ext",
    multiple=True,
    type=str,
    help="File extensions to ignore",
)
@click.option("-u", "--username", type=str, help="Username for remote server")
@click.option("-h", "--host", type=str, help="Hostname of remote server")
@click.option("-p", "--port", default=22, type=int, help="SSH port of remote server")
@click.option("-k", "--key", help="Path to SSH key file")
def sync_directory_cli(
    local_dir="./",
    remote_dir="~/",
    mode="both",
    ignore_dir=[],
    ignore_ext=[],
    username=None,
    host=None,
    port=None,
    key=None,
    sftp=None,
):
    logger.debug(
        f"sync_directory_cli({local_dir}, {remote_dir}, {mode}, {ignore_dir}, {ignore_ext}, {username}, {host}, {port}, {key}, {sftp})"
    )

    local_dir = Path(local_dir)
    remote_dir = Path(remote_dir)

    close_ssh_at_exit = False
    if sftp is None:
        try:
            sftp, ssh = get_client(username, host, port, key)
            close_ssh_at_exit = True
        except Exception as e:
            logger.error(f"Could not get client: {e}")
            return
    elif sftp is not None and any([x is not None for x in [username, host, port, key]]):
        logger.warning(
            "sftp and username, host, port, and key are both not None. Using sftp"
        )

    if mode in ["send", "both"]:
        if remote_dir.name != local_dir.name:
            remote_dir = remote_dir / local_dir.name
        try:
            sftp.stat(str(remote_dir))
        except Exception as e:
            if e.errno == errno.ENOENT:
                logger.info("Creating remote directory %s" % remote_dir)
                sftp.mkdir(str(remote_dir))
            else:
                raise e
        sftp.chdir(str(remote_dir))
        if local_dir.is_file():
            iterdir = [local_dir]
        elif not local_dir.is_dir():
            raise NotADirectoryError(f"{local_dir} is not a directory")
        else:
            iterdir = local_dir.iterdir()
        for f in iterdir:
            if f.is_dir():
                if f.name in ignore_dir:
                    continue
                sync_directory(
                    f, remote_dir / f.name, mode, ignore_dir, ignore_ext, sftp=sftp
                )
            elif f.suffix in ignore_ext:
                continue
            if f.name in sftp.listdir():
                if int(sftp.stat(f.name).st_mtime) < int(f.stat().st_mtime):
                    logger.info(
                        "Putting local path %s to remote path %s"
                        % (f, remote_dir / f.name)
                    )
                    sftp.put(str(f), str(remote_dir / f.name))
                    sftp.utime(f.name, (int(f.stat().st_atime), int(f.stat().st_mtime)))
            else:
                logger.info(
                    "Putting local path %s to remote path %s" % (f, remote_dir / f.name)
                )
                sftp.put(str(f), str(remote_dir / f.name))
                sftp.utime(f.name, (int(f.stat().st_atime), int(f.stat().st_mtime)))

    if mode in ["receive", "both"]:
        sftp.chdir(str(remote_dir))
        if not local_dir.is_dir():
            logger.info("Creating local directory %s" % local_dir)
            local_dir.mkdir()
        for fname, f_attr in zip(sftp.listdir(), sftp.listdir_attr()):
            if stat.S_ISDIR(f_attr.st_mode):
                if fname in ignore_dir:
                    continue
                sync_directory(
                    local_dir / fname,
                    remote_dir / fname,
                    mode,
                    ignore_dir,
                    ignore_ext,
                    sftp=sftp,
                )
            elif Path(fname).suffix in ignore_ext:
                continue
            elif (local_dir / fname).exists():
                if int(f_attr.st_mtime) > int((local_dir / fname).stat().st_mtime):
                    logger.info(
                        "Getting remote path %s to local path %s"
                        % (remote_dir / fname, local_dir / fname)
                    )
                    sftp.get(str(remote_dir / fname), str(local_dir / fname))
                    os.utime(
                        local_dir / fname, (int(f_attr.st_atime), int(f_attr.st_mtime))
                    )

            else:
                logger.info(
                    "Getting remote path %s to local path %s"
                    % (remote_dir / fname, local_dir / fname)
                )
                sftp.get(str(remote_dir / fname), str(local_dir / fname))
                os.utime(
                    local_dir / fname, (int(f_attr.st_atime), int(f_attr.st_mtime))
                )

    if close_ssh_at_exit:
        ssh.close()


@click.command(
    epilog="""Check out the documentation at
               https://pyscope.readthedocs.io/en/latest/
               for more information."""
)
@click.option(
    "-u", "--username", default="telrun", type=str, help="Username for remote server"
)
@click.option("-h", "--host", type=str, help="Hostname of remote server")
@click.option("-p", "--port", default=22, type=int, help="SSH port of remote server")
@click.option(
    "-k",
    "--key",
    default="./config/id_rsa",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to SSH key file",
)
@click.option(
    "-d",
    "--directory",
    nargs=5,
    multiple=True,
    default=[["./", "/home/telrun/", "both", "none", "none"]],
    type=(
        click.Path(exists=True, resolve_path=True, readable=True, writable=True),
        click.Path(),
        str,
        str,
        str,
    ),
    help="""Requires 5 arguments: LOCAL_DIR, REMOTE_DIR, MODE, IGNORE_DIR, IGNORE_EXT.
    LOCAL_DIR is the local directory to sync. REMOTE_DIR is the remote directory to sync to.
    MODE is either 'send', 'receive', or 'both'. IGNORE_DIR is a comma-separated list of
    directories to ignore. IGNORE_EXT is a comma-separated list of file extensions to ignore.
    If IGNORE_DIR or IGNORE_EXT are not needed, use 'none' as the argument. You may specify
    multiple directories to sync by repeating the -d option. Each directory will be placed
    on a single thread.
    """,
)
@click.option(
    "-c",
    "--config",
    default="./config/",
    show_default=True,
    type=click.Path(exists=True, resolve_path=True),
    help="Path to config directory on local machine",
)
@click.option(
    "--do-once",
    "do_once",
    default=False,
    is_flag=True,
    help="Run sync_manager once and exit",
)
@click.version_option()
def sync_manager_cli(
    username=None,
    host=None,
    port=None,
    key=None,
    dirname=None,
    config=None,
    do_once=False,
    do_async=False,
):
    logger.info("Starting sync_manager")
    logger.debug(
        f"""sync_manager_cli(
    username={username},
    host={host},
    port={port},
    key={key},
    dirname={dirname},
    config={config},
    do_async={do_async}
    )"""
    )

    if config is not None:
        config = Path(config)
        if config.is_dir():
            config = config / "sync_manager.cfg"
            if not config.is_file():
                logger.warning(
                    "sync_manager.cfg not found in config directory, using kwargs"
                )

        # Read sync_manager.cfg
        uname, hostname, p, k, directory = read_sync_cfg(config)

    # Override with any arguments passed in
    if username is not None:
        uname = username
    if host is not None:
        hostname = host
    if port is not None:
        p = port
    if key is not None:
        k = key
    if dirname is not None:
        directory = dirname

    if any([x is None for x in [uname, hostname, p, k, directory]]):
        raise ValueError("Missing required arguments")

    # Create thread event
    thread_event = threading.Event()

    # Create clients and threads
    logger.info("Creating ssh/sftp clients and threads")
    threads = []
    sshs = []
    for i, d in enumerate(directory):
        local_dir, remote_dir, mode, ignore_dir, ignore_ext = d
        local_dir = Path(local_dir)
        remote_dir = Path(remote_dir)
        ignore_dir = ignore_dir.split(",") if ignore_dir != "none" else []
        ignore_ext = ignore_ext.split(",") if ignore_ext != "none" else []

        sftp, ssh = get_client(uname, hostname, p, k)
        thread = threading.Thread(
            target=_continuous_sync,
            args=(
                sftp,
                ssh,
                local_dir,
                remote_dir,
                mode,
                ignore_dir,
                ignore_ext,
                thread_event,
            ),
            daemon=True,
            name="thread_%d" % i,
        )
        threads.append(thread)
        sshs.append(ssh)
        thread.start()

    if do_once:
        thread_event.set()
        for t in threads:
            t.join()
        for ssh in sshs:
            ssh.close()
        logger.info("Done")
    elif do_async:
        return thread_event, threads, sshs
    else:
        while True:
            try:
                time.sleep(60)
                click.echo("sync_manager checking in. Press Ctrl+C to gracefully exit")
            except KeyboardInterrupt:
                logger.info("Exiting sync_manager...")
                thread_event.set()
                for t in threads:
                    t.join()
                logger.info("Done")
                break


def read_sync_cfg(fname):
    cfg = configparser.ConfigParser()
    cfg.read(fname)

    uname = cfg.get("DEFAULT", "username", fallback="telrun")
    hostname = cfg["DEFAULT"]["hostname"]
    p = cfg.getint("DEFAULT", "port", fallback=22)
    k = cfg.get("DEFAULT", "key", fallback="./config/id_rsa")

    directory = []
    for section in cfg.sections():
        if section == "DEFAULT":
            continue
        local_dir = cfg[section]["local_dir"]
        remote_dir = cfg[section]["remote_dir"]
        mode = cfg[section].get("mode", fallback="both")
        ignore_dir = cfg[section].get("ignore_dir", fallback="none")
        ignore_ext = cfg[section].get("ignore_ext", fallback="none")
        directory.append((local_dir, remote_dir, mode, ignore_dir, ignore_ext))

    return uname, hostname, p, k, directory


def get_client(username, host, port=22, key=None):
    pw = None
    if key is None:
        pw = click.prompt(f"Enter password for {username}@{host}", hide_input=True)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, key_filename=key, password=pw)
    sftp = ssh.open_sftp()
    return sftp, ssh


def _continuous_sync(
    sftp, ssh, local_dir, remote_dir, mode, ignore_dir, ignore_ext, event
):
    while not event.is_set():
        sync_directory(local_dir, remote_dir, mode, ignore_dir, ignore_ext, sftp=sftp)
        time.sleep(1)
    else:
        logger.info("Thread %s is exiting" % threading.current_thread().name)
        ssh.close()


sync_directory = sync_directory_cli.callback
sync_manager = sync_manager_cli.callback
