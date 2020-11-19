# @Author: Jan Brejcha <janbrejcha>
# @Date:   2020-11-19T16:51:31+01:00
# @Email:  ibrejcha@fit.vutbr.cz, brejchaja@gmail.com
# @Project: Locate
# @Last modified by:   janbrejcha
# @Last modified time: 2020-11-19T20:55:25+01:00

import argparse as ap
import glob
import os
import sys
import cv2
import shutil


def buildArgumentParser():
    parser = ap.ArgumentParser()
    parser.add_argument(
        "input_dir", help="Directory where input dataset \
        definitions written in csv file format are stored.",
        metavar="input-dir"
    )
    parser.add_argument(
        "output_dir", help="Dir where the output dataset csv file will be \
        stored.",
        metavar="output-dir"
    )
    parser.add_argument(
        "--filter", metavar="N", type=int, nargs="+",
        help="Specify which input datasets to load."
    )
    parser.add_argument(
        "--modality", help="Define which modalities shall be \
        used to generate the output dataset. Possible values: photo-photo, \
        photo-render, render-render. Default: photo-photo.",
        default="photo-photo", type=str
    )
    parser.add_argument(
        "--with-images", nargs=2, type=str, help="Set this flag and pass \
        the path where query images are located and the path where database \
        images are located. The images will be automatically converted to jpg\
        and will be consolidated into the output dir so that they can be \
        directly used with the build_image_dataset.py script."
    )
    parser.add_argument(
        "--set-name", help="Define the name of output set, \
        possible options are train, test, index. Default: train.",
        default="train"
    )
    return parser


def loadCSVDatasetPaths(args):
    if args.filter:
        csv_files = []
        for n in args.filter:
            fname = os.path.join(args.input_dir, "*_" + str(n) + ".csv")
            csv_files_p = glob.glob(fname)
            for cf in csv_files_p:
                csv_files.append(cf)
    else:
        csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    return csv_files


def loadAndParseCSVDatasets(args):
    csv_files = loadCSVDatasetPaths(args)
    rows = []
    for csv_file in csv_files:
        with open(csv_file, 'r') as csvf:
            lines = csvf.readlines()
            for line in lines:
                row = line.strip().split(",")
                # strip rows
                row = [r.strip() for r in row]
                # to int
                for r in [1, 2]:
                    row[r] = int(row[r])
                # to float
                for r in [4, 5, 6, 7, 8, 9, 10]:
                    row[r] = float(row[r])
                rows.append(row)
    return rows


def getAllowedTypes(args):
    allowed_types = set()
    modalities = args.modality.split('-')
    re = RuntimeError("Invalid modality selected. Possible values: \
    photo-photo, photo-render, render-render.")
    if len(modalities) != 2:
        raise re
    for modality in modalities:
        if modality == "photo":
            allowed_types.add("query")
        elif modality == "render":
            allowed_types.add("database")
        else:
            raise re
    return allowed_types


def consolidateQueryImage(path, img_name, image_out_dir, class_id, ext=".jpg"):
    image_input_path = os.path.join(path, img_name)
    if not os.path.exists(image_input_path):
        raise RuntimeError(
            "Input query image does not exist at path: "
            + image_input_path
        )

    img_name_parts = os.path.splitext(img_name)
    img_name_base = img_name_parts[0]
    img_name_ext = img_name_parts[1]
    image_output_path = os.path.join(image_out_dir, img_name_base + ext)

    # check whether the output image already exists, if no, we proccess it
    if not os.path.exists(image_output_path):
        if img_name_ext != ext:
            # extension of existing file is not the same as requested
            # we need to load the image and save it in the requested format
            img = cv2.imread(image_input_path)
            cv2.imwrite(image_output_path, img)
        else:
            # the file is in requested format, we just copy it since it is
            # faster than re-encoding
            shutil.copyfile(image_input_path, image_output_path)


def consolidateDatabaseImage(path, img_name, image_out_dir,
                             class_id, ext=".jpg"):
    # database images are either in .png, or in .exr
    curr_ext = ".png"
    # database modality is inferred from the input path, and can be:
    # segments, silhouettes, depth.
    modality_subtype = os.path.basename(path).split("_")[1]
    image_input_path = os.path.join(path, img_name + "_" + modality_subtype + curr_ext)
    if not os.path.exists(image_input_path):
        curr_ext = ".exr"
        image_input_path = os.path.join(path, img_name + curr_ext)
    if not os.path.exists(image_input_path):
        raise RuntimeError(
            "Input database image does not exist at path: "
            + image_input_path
        )

    # database images are defined as tile_x_y/image_id_for_tile
    # to get image id, which is globally unique, we add the tile
    # directory into the image name by replacing '/' with '_'.
    img_out_name = img_name.replace('/', '_')
    image_output_path = os.path.join(image_out_dir, img_out_name + ext)

    # check whether the output image already exists, if no, we proccess it
    if not os.path.exists(image_output_path):
        img = cv2.imread(image_input_path)
        cv2.imwrite(image_output_path, img)


def consolidateImage(args, type, img_name, class_id):
    image_out_dir = os.path.join(args.output_dir, "images", str(class_id))
    if not os.path.exists(image_out_dir):
        os.makedirs(image_out_dir)
    if type == "query":
        consolidateQueryImage(
            args.with_images[0], img_name, image_out_dir, class_id)
    else:
        consolidateDatabaseImage(
            args.with_images[1], img_name, image_out_dir, class_id
        )


def writeGLDV2DatasetCSV(dataset, output_path, setname):
    output_dset_file = os.path.join(output_path, setname + ".csv")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_dset_file, "w") as file:
        # write header
        file.write("id,url,landmark_id\n")
        # write rows
        for row in dataset:
            rowlength = len(row)
            # write cells
            for cidx in range(0, rowlength):
                cell = row[cidx]
                file.write(str(cell))
                # after each cell add comma except the last one
                if cidx < rowlength - 1:
                    file.write(",")
            file.write("\n")


def buildGLDV2Dataset(args):
    dataset = loadAndParseCSVDatasets(args)
    allowed_types = getAllowedTypes(args)
    output_dataset = []
    for row in dataset:
        if row[0] in allowed_types:
            # add this modality to the output
            try:
                if args.with_images:
                    consolidateImage(args, row[0], row[3], row[1])
                img_base = os.path.splitext(row[3])[0]

                # database images are defined as tile_x_y/image_id_for_tile
                # to get image id, which is globally unique, we add the tile
                # directory into the image name by replacing '/' with '_'.
                img_id = img_base.replace('/', '_')

                output_dataset.append([img_id, row[3], row[1]])
            except RuntimeError:
                print(
                    "Unable to process image file " + row[3]
                    + ", file was skipped.", file=sys.stderr
                )

    writeGLDV2DatasetCSV(output_dataset, args.output_dir, args.set_name)


def main():
    parser = buildArgumentParser()
    args = parser.parse_args()
    buildGLDV2Dataset(args)


if __name__ == "__main__":
    main()
