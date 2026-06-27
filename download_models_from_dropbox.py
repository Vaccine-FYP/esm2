import dropbox
import os
import argparse
import tarfile
import json
import requests as req


def extract_tar_gz(file_path, output_directory):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(output_directory)
        print(f"Files extracted successfully to: {output_directory}")


def get_sub_paths(tasks, years, subtypes):
    task_paths = {}
    for task in tasks:
        paths = []
        for year in years:
            for subtype in subtypes:
                paths.append("/flu_%s_%s_%s.tar.gz" % (task, subtype, year))
        task_paths[task] = paths
    return task_paths

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, nargs="+", choices=["lm", "hi"], default=["lm", "hi"])
parser.add_argument('--year', type=str, nargs="+", choices=[str(year) for year in range(2012, 2022)], default=[str(year) for year in range(2012, 2022)])
parser.add_argument('--subtype', type=str, nargs="+", choices=["a_h3n2", "a_h1n1"], default=[ "a_h3n2", "a_h1n1" ])
parser.add_argument('--output_dir', type=str, default="runs")

args = parser.parse_args()
print("Downloading %s models for %s year and for %s subtype" % (args.task, args.year, args.subtype))

dropbox_access_token = input("Please input your dropbox access token: (https://www.dropbox.com/developers/documentation/python#tutorial)")

task2subdir_name = {
    "lm": "flu_lm",
    "hi": "flu_hi_msa_regressor"
}

def download_file_from_dropbox(dropbox_url, download_path):
    task_paths = get_sub_paths(args.task, args.year, args.subtype)
    for task in task_paths:

        task_local_dir = os.path.join(download_path, task2subdir_name[task])
        if not os.path.exists(task_local_dir):
            os.makedirs(task_local_dir)

        for path in task_paths[task]:
            filename = path.strip("/").split("/")[-1]
            local_file = os.path.join(task_local_dir, filename)
            print("Downloading %s to %s" % (filename, local_file))

            headers = {
                "Authorization": "Bearer " + dropbox_access_token,
                "Dropbox-API-Arg": json.dumps({"url": dropbox_url, "path": path})
            }
            r = req.post(
                "https://content.dropboxapi.com/2/sharing/get_shared_link_file",
                headers=headers,
                stream=True
            )
            r.raise_for_status()

            with open(local_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            extract_tar_gz(local_file, task_local_dir)
            os.remove(local_file)

dropbox_file_path = 'https://www.dropbox.com/scl/fo/7d94eqsii2h1jdm5l7mm6/h?rlkey=1n1wafyuapwx5a4c04jc0y7cs&dl=0'

download_file_from_dropbox(dropbox_file_path, args.output_dir)

