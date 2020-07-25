import os


def read_annotation(path):
    """Read an annotation file to get image paths and labels."""
    print(f'Annotation path: {path}, format: ', end='')
    with open(path) as f:
        line = f.readline().strip()
        print('[image path] label')
        content = [l.strip('\n').split(" ", 1) for l in f.readlines() + [line]]
        img_paths, labels = zip(*content)
    dirname = os.path.dirname(path)
    img_paths = [os.path.join(dirname, img_path) for img_path in img_paths]
    print(labels)
    return img_paths, labels


if __name__ == '__main__':
    read_annotation("our_data/annotation.txt")
