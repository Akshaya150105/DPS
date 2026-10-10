"""Download small authentic samples of the Enron and Lucene datasets for local testing."""

import os
import tarfile
import urllib.request


def download_enron_sample(target_dir: str = "datasets/raw/enron", max_files: int = 200) -> int:
    """Stream and extract a small sample of real Enron emails from CMU."""
    url = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
    print(f"Connecting to CMU Enron repository ({url})...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    with urllib.request.urlopen(req, timeout=30) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if member.isfile():
                    # Strip leading 'maildir/' if present so it extracts into target_dir/maildir
                    member_path = member.name
                    # Save under target_dir
                    out_path = os.path.join(target_dir, member_path)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    f = tar.extractfile(member)
                    if f:
                        with open(out_path, "wb") as out_f:
                            out_f.write(f.read())
                        count += 1
                        if count % 50 == 0:
                            print(f"  Extracted {count}/{max_files} Enron emails...")
                        if count >= max_files:
                            break

    print(f"Successfully extracted {count} Enron emails to '{target_dir}'.")
    return count


def download_lucene_sample(target_dir: str = "datasets/raw/lucene") -> str:
    """Download a real monthly mbox archive from Apache Lucene developer list."""
    url = "https://mail-archives.apache.org/mod_mbox/lucene-dev/200201.mbox"
    print(f"Downloading Lucene mbox from {url}...")
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, "lucene-dev-200201.mbox")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)

    print(f"Successfully saved {len(data)} bytes to '{out_path}'.")
    return out_path


def main() -> None:
    print("=== Downloading Enron Sample ===")
    download_enron_sample(max_files=200)
    print("\n=== Downloading Lucene Sample ===")
    download_lucene_sample()
    print("\nDone! Ready for preprocess.py.")


if __name__ == "__main__":
    main()
