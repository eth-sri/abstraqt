import argparse
import os
import shutil

from appdirs import user_cache_dir

cache_dir = user_cache_dir('abstraqt')
os.makedirs(cache_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser('Handle cache directory')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--print', action='store_true')
    args = parser.parse_args()

    if args.clean:
        i = 1
        while True:
            backup_directory = cache_dir + '.bak.' + str(i)
            if not os.path.exists(backup_directory):
                print('Moving cache directory', cache_dir, 'to', backup_directory)
                print()

                shutil.move(cache_dir, backup_directory)

                break

            i += 1

    if args.print:
        print(cache_dir)


if __name__ == "__main__":
    main()
