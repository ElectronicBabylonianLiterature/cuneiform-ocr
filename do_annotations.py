import json
import os
from pymongo import MongoClient
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv
import gridfs
from PIL import Image, UnidentifiedImageError
import io
from tqdm import tqdm
from datetime import datetime
from data_processing.divide_photos import divide_tablet_photo

# Allow large image processing
Image.MAX_IMAGE_PIXELS = None

# get thresholds and tag from environment variables
tag = os.getenv('TAG', '0')
THRESHOLD_CERTAIN = float(os.getenv('THRESHOLD', '0.8'))
Y_THRESHOLD = int(os.getenv('Y_THRESHOLD', '35'))

if __name__ == "__main__":
    client = MongoClient("YOUR_MONGODB_URI")
    db = client['ebl']
    files_collection = db['photos.files']
    chunks_collection = db['photos.chunks']
    done_fragments = {}
    fs = gridfs.GridFS(db, collection='photos')
    classes = ['ABZ58', 'ABZ441', 'ABZ207', 'ABZ55', 'ABZ139', 'ABZ597', 'ABZ343', 'ABZ142', 'ABZ73', 'ABZ59', 'ABZ586', 'ABZ579', 'ABZ457', 'ABZ427', 'ABZ86', 'ABZ212', 'ABZ5', 'ABZ537', 'ABZ376', 'ABZ335', 'ABZ170', 'ABZ342', 'ABZ324', 'ABZ480', 'ABZ61', 'ABZ206', 'ABZ545', 'ABZ99', 'ABZ72', 'ABZ112', 'ABZ142a', 'ABZ396', 'ABZ103', 'ABZ13', 'ABZ70', 'ABZ69', 'ABZ437', 'ABZ381', 'X', 'ABZ279', 'ABZ52', 'ABZ128', 'ABZ97', 'ABZ151', 'ABZ465', 'ABZ461', 'ABZ595', 'ABZ468', 'ABZ1', 'ABZ449', 'ABZ318', 'ABZ384', 'ABZ214', 'ABZ111', 'ABZ367', 'ABZ84', 'ABZ319', 'ABZ62', 'ABZ314', 'ABZ556', 'ABZ7', 'ABZ230', 'ABZ74', 'ABZ144', 'ABZ331', 'ABZ330', 'ABZ598a', 'ABZ575', 'ABZ322', 'NoABZ0', 'ABZ6', 'ABZ354', 'ABZ172', 'ABZ399', 'ABZ328', 'ABZ471', 'ABZ332', 'ABZ593', 'ABZ233', 'ABZ148', 'ABZ538', 'ABZ12', 'ABZ57', 'ABZ481', 'ABZ313', 'ABZ167', 'ABZ15', 'ABZ68', 'ABZ353', 'ABZ398', 'ABZ532', 'ABZ371', 'ABZ231', 'ABZ80', 'ABZ314', 'ABZ295', 'ABZ115', 'ABZ411', 'ABZ308', 'ABZ191', 'ABZ296', 'ABZ412', 'ABZ565', 'ABZ401', 'ABZ589', 'ABZ211', 'ABZ472', 'ABZ570', 'ABZ79', 'ABZ75', 'ABZ298', 'ABZ420', 'ABZ535', 'ABZ134', 'ABZ536', 'ABZ101', 'ABZ533', 'ABZ536', 'ABZ126', 'ABZ94', 'ABZ9', 'ABZ232', 'ABZ393', 'ABZ60', 'ABZ104', 'ABZ131', 'ABZ306', 'ABZ38', 'ABZ470', 'ABZ557', 'ABZ333', 'NoABZ0', 'ABZ147', 'ABZ145', 'ABZ56', 'ABZ564', 'ABZ383', 'ABZ360', 'ABZ114', 'ABZ138', 'ABZ331e+152i', 'ABZ297', 'ABZ334', 'ABZ366', 'ABZ50', 'ABZ455', 'ABZ598b', 'ABZ339', 'ABZ205', 'ABZ78', 'ABZ87', 'ABZ554', 'ABZ85', 'ABZ536', 'ABZ312', 'ABZ69', 'ABZ433', 'ABZ124', 'ABZ164', 'ABZ129a', 'NoABZ0', 'ABZ76', 'ABZ326', 'ABZ143', 'ABZ440', 'ABZ559', 'ABZ307', 'ABZ374', 'ABZ74', 'ABZ451', 'ABZ574', 'NoABZ0', 'ABZ529']
    with open('output.json', 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    for i in output_data:
        done_fragments[i['filename']] = True
    photo_count = files_collection.count_documents({})
    cursor = files_collection.find({})
    config_file = 'configs/detr.py'
    checkpoint_file = '~/erc-work-data/retrained_models/detr-173/epoch_1000.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    register_all_modules()
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    date_str = datetime.now().strftime("%m-%d")
    output_file_name = f"output_inference_{date_str}_{tag}.json"

    info = "tag {}, threshold {}, y_threshold {}".format(tag, THRESHOLD_CERTAIN, Y_THRESHOLD)
    print("starting inference with " + info)

    count = 0
    checkpoints = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

    for doc in tqdm(cursor, total=photo_count):
        try:
            file_id = doc['_id']
            filename = doc['filename']
            if filename.endswith('.tif'):
                continue
            if filename in done_fragments:
                print(f"Skipping file {filename}: Already processed")
                continue
            # Retrieve the binary data from GridFS
            binary_data = fs.get(file_id).read()

            # Convert binary data to image and save as PNG
            try:
                image = Image.open(io.BytesIO(binary_data))
                filename_save = tag + filename
                image.save(f"{filename_save}")
            except UnidentifiedImageError as e:
                print(f"Skipping file {filename}: Image cannot be identified - {str(e)}")
                continue  # Skip to the next file if the image is invalid

            img = mmcv.imread(f"{filename_save}")
            cropped_images, crop_coordinates = divide_tablet_photo(img, visualize=False, logging=False, return_coordinates=True)

            def line_signs(sorted_indexed_bounding_boxes, labels, classes):
                lines = []
                current_line = []
                y_threshold = Y_THRESHOLD

                for i, (index, box) in enumerate(sorted_indexed_bounding_boxes):
                    if not current_line:
                        current_line.append((index, box))
                    else:
                        prev_box = current_line[-1][1]
                        if box[1] - prev_box[1] < y_threshold:
                            current_line.append((index, box))
                        else:
                            lines.append(current_line)
                            current_line = [(index, box)]
                if current_line:
                    lines.append(current_line)

                sorted_lines = [sorted(line, key=lambda item: item[1][0]) for line in lines]

                bounding_boxes = []
                result = ''
                for line in sorted_lines:
                    for index, box in line:
                        if classes[labels[index]] == 'NoABZ0':
                            result += 'X '
                        else:
                            result += classes[labels[index]] + ' '
                        bounding_boxes.append(box.tolist())  # Convert tensor to list for easier handling
                    result += '\n'
                return result, bounding_boxes

            line_signs_results = ''
            bounding_boxes = []
            for idx, img_piece in enumerate(cropped_images):
                # Perform inference
                result = inference_detector(model, img_piece)
                OCR_result = result.pred_instances.cpu()
                labels, bboxes = OCR_result['labels'], OCR_result['bboxes']
                certain_scores_idx = len(OCR_result['scores'][OCR_result['scores'] > THRESHOLD_CERTAIN])
                certain_scores_idx
                certain_bboxes = bboxes[:certain_scores_idx]
                labels = OCR_result['labels'][:certain_scores_idx]

                # Sorting bounding boxes and grouping them into lines
                indexed_bounding_boxes = list(enumerate(certain_bboxes))
                sorted_indexed_bounding_boxes = sorted(indexed_bounding_boxes, key=lambda item: (item[1][1], item[1][0]))
                lined_signs, bounding_boxes_of_one_piece = line_signs(sorted_indexed_bounding_boxes, labels, classes)

                # Convert bounding boxes from small piece coordinates to original image coordinates
                piece_offset_x = crop_coordinates[idx]['x']
                piece_offset_y = crop_coordinates[idx]['y']

                # Transform each bbox to original image coordinates
                transformed_bboxes = []
                for bbox in bounding_boxes_of_one_piece:
                    # bbox format is [x1, y1, x2, y2]
                    transformed_bbox = [
                        bbox[0] + piece_offset_x,  # x1
                        bbox[1] + piece_offset_y,  # y1
                        bbox[2] + piece_offset_x,  # x2
                        bbox[3] + piece_offset_y   # y2
                    ]
                    transformed_bboxes.append(transformed_bbox)

                line_signs_results += lined_signs
                bounding_boxes.extend(transformed_bboxes)

            output_data.append({
                "ocredSigns": line_signs_results,
                "filename": filename,
                "ocredSignsCoordinates": bounding_boxes
            })
            os.remove(filename_save)
            count += 1
            if count in checkpoints:
                with open(output_file_name, "w") as json_file:
                    json.dump(output_data, json_file, indent=4)
                print(f"Checkpoint: Processed {count} files, saved to {output_file_name} " +  info)

        except Exception as e:
            print(f"Skipping file {filename} due to error: {str(e)}")
            continue  # Skip the file and move to the next one in case of any error

    with open(output_file_name, "w") as json_file:
        json.dump(output_data, json_file, indent=4)
    print(f"Finished processing. Total files processed: {count}. Results saved to {output_file_name} " + info)
