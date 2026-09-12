from pathlib import Path

base = Path("dataset/eggplant_dataset")

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")
    for folder in sorted((base / split).iterdir()):
        if folder.is_dir():
            count = len([
                f for f in folder.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
            ])
            print(folder.name, ":", count)
        from pathlib import Path
        import hashlib

        BASE = Path("dataset/eggplant")
        splits = ["train", "val", "test"]

        hashes = {}

        for split in splits:
            for file in (BASE / split).rglob("*"):
                if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    h = hashlib.md5(file.read_bytes()).hexdigest()
                    hashes.setdefault(h, []).append((split, str(file)))

        duplicate_count = 0

        for h, files in hashes.items():
            split_names = {x[0] for x in files}

            if len(split_names) > 1:
                duplicate_count += 1
                print("\nDUPLICATE ACROSS SPLITS:")
                for split, path in files:
                    print(split, "->", path)

        print("\nTotal cross-split duplicate groups:", duplicate_count)