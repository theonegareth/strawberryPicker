from pathlib import Path

class_name = 'strawberry'

def rewrite_single_class_data_yaml(dataset_dir, class_name='strawberry'):
    dataset_dir = Path(dataset_dir)
    data_yaml_path = dataset_dir / 'data.yaml'
    if not data_yaml_path.exists():
        print('⚠️  data.yaml not found, skipping rewrite.')
        return

    train_path = dataset_dir / 'train' / 'images'
    val_path = dataset_dir / 'valid' / 'images'
    test_path = dataset_dir / 'test' / 'images'

    content_lines = [
        '# Strawberry-only dataset',
        f'train: {train_path}',
        f'val: {val_path}',
        f"test: {test_path if test_path.exists() else ''}",
        '',
        'nc: 1',
        f"names: ['{class_name}']",
    ]

    data_yaml_path.write_text('\n'.join(content_lines) + '\n')
    print(f"✅ data.yaml updated for single-class training ({class_name}).")


def enforce_single_class_dataset(dataset_dir, target_class=0, class_name='strawberry'):
    dataset_dir = Path(dataset_dir)
    stats = {'split_kept': {}, 'labels_removed': 0, 'images_removed': 0}
    allowed_ext = ['.jpg', '.jpeg', '.png']

    for split in ['train', 'valid', 'test']:
        labels_dir = dataset_dir / split / 'labels'
        images_dir = dataset_dir / split / 'images'
        if not labels_dir.exists():
            continue

        kept = 0
        for label_path in labels_dir.glob('*.txt'):
            kept_lines = []
            for raw_line in label_path.read_text().splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                if class_id == target_class:
                    kept_lines.append(line)

            if kept_lines:
                label_path.write_text('\n'.join(kept_lines) + '\n')
                kept += len(kept_lines)
            else:
                label_path.unlink()
                stats['labels_removed'] += 1
                for ext in allowed_ext:
                    candidate = images_dir / f"{label_path.stem}{ext}"
                    if candidate.exists():
                        candidate.unlink()
                        stats['images_removed'] += 1
                        break

        stats['split_kept'][split] = kept

    rewrite_single_class_data_yaml(dataset_dir, class_name)

    print('\n🍓 Strawberry-only filtering summary:')
    for split, count in stats['split_kept'].items():
        print(f"   {split}: {count} annotations kept")
    print(f"   Label files removed: {stats['labels_removed']}")
    print(f"   Images removed (non-strawberry or empty labels): {stats['images_removed']}")

    return stats


if __name__ == "__main__":
    # Example usage - replace with your dataset path
    dataset_path = "path/to/your/dataset"  # Update this path

    if dataset_path and Path(dataset_path).exists():
        strawberry_stats = enforce_single_class_dataset(dataset_path, target_class=0, class_name=class_name)
    else:
        print("Please set a valid dataset_path variable.")