#!/usr/bin/env python3
"""
Convert a CVAT for Images 1.1 XML annotation file to COCO 1.0 JSON format.

All images are included in the output, including unannotated background images
which are tagged as 'background'. Export your annotation project from CVAT using
'CVAT for Images 1.1' before running this script.
"""

import os
import argparse
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

def cvat_to_coco(xml_path: str, output_path: str):
    """
    Converts a CVAT for Images 1.1 XML file to COCO 1.0 JSON format.

    This script ensures that all images are included in the output, even those
    without any annotations (background/negative samples).
    """
    print(f"Parsing CVAT XML file: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Initialize COCO structure
    coco_data = {
        "licenses": [],
        "info": {"description": "Converted from CVAT XML to COCO JSON"},
        "categories": [
            # The training script determines occluded vs. non-occluded,
            # so we only need one category here.
            {"supercategory": "animal", "id": 1, "name": "chimp"}
        ],
        "images": [],
        "annotations": []
    }

    annotation_id_counter = 1

    print("Converting annotations...")
    for image_elem in tqdm(root.findall('image')):
        image_id = int(image_elem.get('id'))
        image_filename = image_elem.get('name')
        width = int(image_elem.get('width'))
        height = int(image_elem.get('height'))

        # Add image entry for every image
        image_entry = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_filename,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        
        # Process annotations for this image
        num_annotations_for_image = 0
        for box_elem in image_elem.findall('box'):
            label = box_elem.get('label')
            if label.lower() != 'chimp' and label.lower() != 'chimpanzee':
                continue # Skip labels we don't care about

            xtl = float(box_elem.get('xtl'))
            ytl = float(box_elem.get('ytl'))
            xbr = float(box_elem.get('xbr'))
            ybr = float(box_elem.get('ybr'))

            # CVAT's 'occluded' is a 0 or 1 string.
            # Our train script expects a boolean attribute.
            is_occluded = box_elem.get('occluded') == '1'

            # Convert bbox format from [xtl, ytl, xbr, ybr] to COCO's [x, y, width, height]
            bbox_x = xtl
            bbox_y = ytl
            bbox_w = xbr - xtl
            bbox_h = ybr - ytl

            coco_data['annotations'].append({
                "id": annotation_id_counter,
                "image_id": image_id,
                "category_id": 1, # "chimp"
                "segmentation": [],
                "area": bbox_w * bbox_h,
                "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                "iscrowd": 0,
                "attributes": {"occluded": is_occluded}
            })
            annotation_id_counter += 1
            num_annotations_for_image += 1

        # If the image has no valid annotations, tag it as background
        if num_annotations_for_image == 0:
            image_entry["tags"] = ["background"]

        coco_data['images'].append(image_entry)

    # Save the merged file
    print(f"\nConversion complete.")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"COCO JSON file saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a CVAT for Images 1.1 XML file to COCO 1.0 JSON format."
    )
    parser.add_argument(
        '--xml_input',
        type=str,
        required=True,
        help="Path to the input CVAT XML file."
    )
    parser.add_argument(
        '--json_output',
        type=str,
        default='data/annotations/annotations.json',
        help="Path to save the output COCO JSON file (default: data/annotations/annotations.json)."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.xml_input):
        print(f"Error: Input XML file not found at '{args.xml_input}'")
        print("Export your annotation project from CVAT using 'CVAT for images 1.1' format.")
    else:
        cvat_to_coco(args.xml_input, args.json_output)
