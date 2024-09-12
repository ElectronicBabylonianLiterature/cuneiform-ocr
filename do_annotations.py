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

# Allow large image processing
Image.MAX_IMAGE_PIXELS = None

if __name__ == "__main__":
    client = MongoClient('%YOURMONGODBCONNECTION%)
    db = client['ebl']
    files_collection = db['photos.files']
    chunks_collection = db['photos.chunks']
    done_fragments = {}
    fs = gridfs.GridFS(db, collection='photos')
    classes = ['ABZ533', 'ABZ61', 'ABZ308', 'ABZ342', 'ABZ55', 'ABZ384', 'ABZ68', 'ABZ15', 'ABZ335', 'ABZ579', 'ABZ331e+152i', 'ABZ139', 'ABZ401', 'ABZ142', 'ABZ86', 'ABZ537', 'ABZ111', 'ABZ457', 'ABZ99', 'ABZ461', 'ABZ80', 'ABZ70', 'ABZ381', 'ABZ306', 'ABZ371', 'ABZ449', 'ABZ366', 'ABZ318', 'ABZ112', 'ABZ5', 'ABZ85', 'ABZ440', 'ABZ427', 'ABZ396', 'ABZ354', 'ABZ104', 'ABZ231', 'ABZ545', 'ABZ75', 'ABZ214', 'ABZ13', 'ABZ589', 'ABZ328', 'ABZ343', 'ABZ367', 'ABZ232', 'ABZ191', 'ABZ62', 'ABZ1', 'ABZ142a', 'ABZ411', 'ABZ151', 'ABZ554', 'ABZ147', 'ABZ319', 'ABZ206', 'NoABZ0', 'ABZ73', 'ABZ480', 'ABZ97', 'ABZ597', 'ABZ312', 'ABZ465', 'ABZ334', 'ABZ470', 'ABZ532', 'ABZ212', 'ABZ536', 'ABZ52', 'ABZ12', 'ABZ330', 'ABZ84', 'ABZ69', 'ABZ72', 'ABZ331', 'ABZ437', 'ABZ230', 'ABZ397', 'ABZ279', 'ABZ6', 'ABZ60', 'ABZ74', 'ABZ144', 'ABZ383', 'ABZ296', 'ABZ211', 'ABZ441', 'ABZ58', 'ABZ128', 'ABZ570', 'ABZ586', 'ABZ207', 'ABZ324', 'ABZ455', 'ABZ376', 'ABZ78', 'ABZ471', 'ABZ535', 'ABZ295', 'ABZ59', 'ABZ314', 'ABZ145', 'ABZ353', 'ABZ412', 'ABZ575', 'ABZ115', 'ABZ7', 'ABZ38', 'ABZ472', 'ABZ101', 'ABZ167', 'ABZ322', 'ABZ172', 'ABZ79', 'ABZ468', 'ABZ595', 'ABZ399', 'ABZ313', 'ABZ529', 'ABZ143', 'ABZ148', 'ABZ339', 'ABZ134', 'ABZ2', 'ABZ538', 'ABZ393', 'ABZ298', 'ABZ50', 'ABZ483', 'ABZ559', 'ABZ87', 'ABZ94', 'ABZ152', 'ABZ138', 'ABZ164', 'ABZ565', 'ABZ205', 'ABZ598a', 'ABZ307', 'ABZ9', 'ABZ398', 'ABZ126', 'ABZ124', 'ABZ195', 'ABZ131', 'ABZ375', 'ABZ56', 'ABZ556', 'ABZ170']
    with open('output.json', 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    for i in output_data:
        done_fragments[i['filename']] = True
    cursor = files_collection.find({})
    config_file = 'checkpoint_config/detr.py'
    checkpoint_file = 'checkpoint_config/epoch_600.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    register_all_modules()
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    for doc in tqdm(cursor):
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
                image.save(f"{filename}")
            except UnidentifiedImageError as e:
                print(f"Skipping file {filename}: Image cannot be identified - {str(e)}")
                continue  # Skip to the next file if the image is invalid

            img = mmcv.imread(f"{filename}")

            # Perform inference
            result = inference_detector(model, img)
            OCR_result = result.pred_instances.cpu()
            labels, bboxes = OCR_result['labels'], OCR_result['bboxes']
            THRESHOLD_CERTAIN = 0.5
            certain_scores_idx = len(OCR_result['scores'][OCR_result['scores'] > THRESHOLD_CERTAIN])
            certain_scores_idx
            certain_bboxes = bboxes[:certain_scores_idx]
            labels = OCR_result['labels'][:certain_scores_idx]

            # Sorting bounding boxes and grouping them into lines
            indexed_bounding_boxes = list(enumerate(certain_bboxes))
            sorted_indexed_bounding_boxes = sorted(indexed_bounding_boxes, key=lambda item: (item[1][1], item[1][0]))
            lines = []
            current_line = []
            y_threshold = 50
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

            # Collect OCR result
            result = ""
            bounding_boxes = []
            for line in sorted_lines:
                for index, box in line:
                    if classes[labels[index]] == 'NoABZ0':
                        result += 'X '
                    else:
                        result += classes[labels[index]] + ' '
                    bounding_boxes.append(box.tolist())  # Convert tensor to list for easier handling
                result += '\n'

            output_data.append({
                "ocredSigns": result,
                "filename": filename,
                "ocredSignsCoordinates": bounding_boxes
            })
            os.remove(filename)
        
        except Exception as e:
            print(f"Skipping file {filename} due to error: {str(e)}")
            continue  # Skip the file and move to the next one in case of any error

    with open("output_new.json", "w") as json_file:
        json.dump(output_data, json_file, indent=4)
