"""
Download Laboro Tomato (1.6 GB) and tomatOD (170 MB) into data/raw/.

Both CC BY-NC-SA 4.0, so the repo keeps the code that rebuilds them, not the
pixels. Resumable, because the Laboro host drops the connection partway through
often enough that starting over is not an option.

Usage: python cross_dataset/fetch.py
"""
import atexit
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
ARCHIVES = RAW / "_archives"

SOURCES = {
    "laboro_tomato": [
        "http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato.zip",
    ],
    "tomatod": [
        "https://datasets-u2m.s3.eu-west-3.amazonaws.com/tomatOD_images.zip",
        "https://datasets-u2m.s3.eu-west-3.amazonaws.com/tomatOD_annotations.zip",
    ],
}


def remote_size(url):
    """Content-Length, or None if the server will not say."""
    out = subprocess.run(["curl", "-sIL", "--max-time", "30", url],
                         capture_output=True, text=True).stdout
    sizes = [line.split()[1].strip() for line in out.splitlines()
             if line.lower().startswith("content-length:")]
    return int(sizes[-1]) if sizes else None


def download(url, dst, attempts=10):
    """Fetch url to dst, resuming until the bytes are all there.

    Completeness is decided by comparing sizes, not by one curl's exit status -
    an attempt died at 1.32 GB with curl 56 and reported nothing useful.
    --speed-limit/--speed-time matter as much as the retries: a connection that
    goes quiet but stays open hangs forever and no retry ever fires."""
    expected = remote_size(url)
    have = dst.stat().st_size if dst.exists() else 0

    if expected and have == expected:
        print(f"have     : {dst.name} ({have / 1e6:.0f} MB, complete)")
        return

    for attempt in range(1, attempts + 1):
        if have:
            print(f"resuming : {dst.name} from {have / 1e6:.0f} MB"
                  + (f" of {expected / 1e6:.0f} MB" if expected else "")
                  + f"  (attempt {attempt}/{attempts})")
        else:
            print(f"fetching : {url}  (attempt {attempt}/{attempts})")

        code = subprocess.run(
            ["curl", "-fL", "-C", "-",
             "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
             # abort rather than hang if the stream drops below 1 KB/s for a minute
             "--speed-limit", "1024", "--speed-time", "60",
             "-o", str(dst), url],
        ).returncode

        have = dst.stat().st_size if dst.exists() else 0
        if code == 0 and (expected is None or have == expected):
            print(f"done     : {dst.name} ({have / 1e6:.0f} MB)")
            return
        if expected and have >= expected:
            return
        print(f"  curl exited {code}, have {have / 1e6:.0f} MB - retrying")

    raise RuntimeError(
        f"{dst.name}: stuck at {have / 1e6:.0f} MB"
        + (f" of {expected / 1e6:.0f} MB" if expected else "")
        + f" after {attempts} attempts"
    )


def extract(archive, dst_dir):
    # a truncated download only fails when something reads it, so check here
    # rather than two steps later as a confusing missing-images error
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(
            f"{archive.name} is not a valid zip ({archive.stat().st_size / 1e6:.0f} MB) "
            f"- the download is incomplete; delete it and re-run"
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = z.namelist()
        # already unpacked; the Laboro zip takes a while and re-runs should be cheap
        sample = next((m for m in members if not m.endswith("/")), None)
        if sample and (dst_dir / sample).exists():
            print(f"unpacked : {archive.name} -> {dst_dir.name}")
            return
        print(f"extract  : {archive.name} ({len(members)} entries)")
        z.extractall(dst_dir)


def acquire_lock():
    """Refuse to run while another copy is downloading.

    Two `curl -C -` on the same archive corrupt it silently: both append, the
    bytes interleave, the file ends up the right size and only fails at unzip."""
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    lock = ARCHIVES / ".fetch.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        owner = lock.read_text().strip()
        if owner.isdigit() and _alive(int(owner)):
            raise SystemExit(
                f"another fetch.py is running (pid {owner}); wait for it to finish. "
                f"If it is gone, delete {lock}"
            )
        print(f"clearing stale lock from pid {owner}")
        lock.unlink()
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    atexit.register(lambda: lock.exists() and lock.unlink())


def _alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError) as err:
        return isinstance(err, PermissionError)
    return True


def main():
    acquire_lock()

    for name, urls in SOURCES.items():
        dst_dir = RAW / name
        for url in urls:
            archive = ARCHIVES / url.rsplit("/", 1)[-1]
            download(url, archive)
            extract(archive, dst_dir)

    print("\nlayout under", RAW)
    for p in sorted(RAW.rglob("*")):
        depth = len(p.relative_to(RAW).parts)
        if depth <= 3 and p.is_dir() and "_archives" not in p.parts:
            n = sum(1 for _ in p.glob("*"))
            print(f"  {'  ' * (depth - 1)}{p.name}/  ({n} entries)")


if __name__ == "__main__":
    sys.exit(main())
