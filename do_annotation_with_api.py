import json
import os
from pymongo import MongoClient
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time
import requests
import cv2
import numpy as np
from matplotlib import pyplot as plt

classes = ['ABZ533', 'ABZ61', 'ABZ308', 'ABZ342', 'ABZ55', 'ABZ384', 'ABZ68', 'ABZ15', 'ABZ335', 'ABZ579', 'ABZ331e+152i', 'ABZ139', 'ABZ401', 'ABZ142', 'ABZ86', 'ABZ537', 'ABZ111', 'ABZ457', 'ABZ99', 'ABZ461', 'ABZ80', 'ABZ70', 'ABZ381', 'ABZ306', 'ABZ371', 'ABZ449', 'ABZ366', 'ABZ318', 'ABZ112', 'ABZ5', 'ABZ85', 'ABZ440', 'ABZ427', 'ABZ396', 'ABZ354', 'ABZ104', 'ABZ231', 'ABZ545', 'ABZ75', 'ABZ214', 'ABZ13', 'ABZ589', 'ABZ328', 'ABZ343', 'ABZ367', 'ABZ232', 'ABZ191', 'ABZ62', 'ABZ1', 'ABZ142a', 'ABZ411', 'ABZ151', 'ABZ554', 'ABZ147', 'ABZ319', 'ABZ206', 'NoABZ0', 'ABZ73', 'ABZ480', 'ABZ97', 'ABZ597', 'ABZ312', 'ABZ465', 'ABZ334', 'ABZ470', 'ABZ532', 'ABZ212', 'ABZ536', 'ABZ52', 'ABZ12', 'ABZ330', 'ABZ84', 'ABZ69', 'ABZ72', 'ABZ331', 'ABZ437', 'ABZ230', 'ABZ397', 'ABZ279', 'ABZ6', 'ABZ60', 'ABZ74', 'ABZ144', 'ABZ383', 'ABZ296', 'ABZ211', 'ABZ441', 'ABZ58', 'ABZ128', 'ABZ570', 'ABZ586', 'ABZ207', 'ABZ324', 'ABZ455', 'ABZ376', 'ABZ78', 'ABZ471', 'ABZ535', 'ABZ295', 'ABZ59', 'ABZ314', 'ABZ145', 'ABZ353', 'ABZ412', 'ABZ575', 'ABZ115', 'ABZ7', 'ABZ38', 'ABZ472', 'ABZ101', 'ABZ167', 'ABZ322', 'ABZ172', 'ABZ79', 'ABZ468', 'ABZ595', 'ABZ399', 'ABZ313', 'ABZ529', 'ABZ143', 'ABZ148', 'ABZ339', 'ABZ134', 'ABZ2', 'ABZ538', 'ABZ393', 'ABZ298', 'ABZ50', 'ABZ483', 'ABZ559', 'ABZ87', 'ABZ94', 'ABZ152', 'ABZ138', 'ABZ164', 'ABZ565', 'ABZ205', 'ABZ598a', 'ABZ307', 'ABZ9', 'ABZ398', 'ABZ126', 'ABZ124', 'ABZ195', 'ABZ131', 'ABZ375', 'ABZ56', 'ABZ556', 'ABZ170']

class Fragment:
    def __init__(self, id="", filename=""):
        if filename != "":
            id = os.path.splitext(os.path.basename(filename))[0]
        self.id = id
        self.filename = filename

        self.bbox = None 
        self.label = None
        self.score = None
        self.signs_gt = ""
        self.orced_signs = ""
        self.image = None
    
    def set_detection(self, bbox, label, score):
        self.bbox = bbox
        self.label = label
        self.score = score

    def set_signs_gt(self, signs):
        self.signs_gt = signs

    def set_orced_signs(self, signs):
        self.orced_signs = signs

    def set_inference_result(self, result):
        self.result = result

    def visualize_with_result(self, visualizer):
        visualizer.add_datasample(
            'result',
            self.image,
            data_sample=self.result,
            draw_gt=False,
            wait_time=0,
        )
        visualizer.show()
        visualizer.fig_save
    
    def visualize_with_bbox(self, img):
        if self.bbox is not None:
            # Make a copy to avoid modifying original image
            img_draw = img.copy()
            # Draw each bbox (assume bbox format: [x1, y1, x2, y2])
            color = (0, 255, 0)  # green
            thickness = 2
            if isinstance(self.bbox, list):
                bboxes = self.bbox
            else:
                bboxes = self.bbox.tolist() if hasattr(self.bbox, 'tolist') else self.bbox
            for box in bboxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
            # Show image with matplotlib
            plt.figure(figsize=(10, 10))
            plt.imshow(img_draw)
            plt.axis('off')
            plt.show()

    # print fragment info
    def __repr__(self):
        return f"Fragment(id={self.id}, filename={self.filename}, bbox={self.bbox}, label={self.label}, score={self.score}, signs_gt={repr(self.signs_gt)})"

class EBLApiClient:
    BASE_URL = "https://ebl.badw.de/api/fragments/"

    def get_fragment_properties(self, fragment_id):
        url = f"{self.BASE_URL}{fragment_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def get_signs(self, fragment_id):
        fragment = self.get_fragment_properties(fragment_id)
        return fragment.get("signs", [])
    
    def get_photo(self, fragment_id):
        url = f"{self.BASE_URL}{fragment_id}/photo"
        response = requests.get(url)
        response.raise_for_status()
        photo_data = response.content
        return photo_data

# Example usage of client:
client = EBLApiClient()
f = Fragment("BM.40757")
signs = client.get_signs(f.id)
photo_data = client.get_photo(f.id)
f.set_signs_gt(signs)
with open("BM.40757_photo.jpg", "wb") as photo_file:
    photo_file.write(photo_data)
    
# Read Photo File List
with open('eBL__photos.json', 'r') as f:
    photo_files = json.load(f)
elb_photo_fragments = []
for photo_file in photo_files:
    # Process each photo file
    fragment = Fragment(filename=photo_file['filename'])
    elb_photo_fragments.append(fragment)
print(f"Found {len(elb_photo_fragments)} photo names")

NUM_TO_PROCESS = 20

fragments_to_process = elb_photo_fragments[:NUM_TO_PROCESS]

# Load model
config_file = 'checkpoint_config/detr.py'
checkpoint_file = 'checkpoint_config/epoch_1000.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
register_all_modules()
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

def download_fragment_data(fragment: Fragment, client):
    fragment_id = fragment.id
    try:
        signs = client.get_signs(fragment_id)
        photo_data = client.get_photo(fragment_id)
        fragment.set_signs_gt(signs)
        fragment.image = mmcv.imfrombytes(photo_data, channel_order='rgb')
        return fragment
    except Exception as e:
        print(f"Error downloading fragment {fragment_id}: {e}")
        return None

def process_fragment_data(fragment: Fragment, model, classes):
    try:
        # Process image
        img = fragment.image
        result = inference_detector(model, img)
        fragment.set_inference_result(result)

        OCR_result = result.pred_instances.cpu()
        labels, bboxes = OCR_result['labels'], OCR_result['bboxes']
        THRESHOLD_CERTAIN = 0.5
        certain_scores_idx = len(OCR_result['scores'][OCR_result['scores'] > THRESHOLD_CERTAIN])
        certain_bboxes = bboxes[:certain_scores_idx]
        labels = OCR_result['labels'][:certain_scores_idx]

        # Sort and group bounding boxes into lines
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

        # Collect OCR results
        ocr_text = ""
        bounding_boxes = []
        for line in sorted_lines:
            for index, box in line:
                if classes[labels[index]] == 'NoABZ0':
                    ocr_text += 'X '
                else:
                    ocr_text += classes[labels[index]] + ' '
                bounding_boxes.append(box.tolist())
            ocr_text += '\n'

        filename = fragment.filename
        fragment.set_orced_signs(ocr_text.strip())
        fragment.set_detection(bbox=bounding_boxes, label=labels, score=None)
        return {
            "ocredSigns": ocr_text,
            "filename": filename,
            "ocredSignsCoordinates": bounding_boxes
        }
    
    except Exception as e:
        print(f"Error processing fragment {fragment.id}: {e}")
        return None


# # test single fragments
# filename = "BM.40757.jpg"
# fragment = Fragment(filename=filename)
# ok = download_fragment_data(fragment, client)
# if ok:
#     test_result = process_fragment_data(fragment, model, classes)
#     print(f"Processed {fragment.id}:\n{fragment.orced_signs if test_result else 'Failed'}")
# else:
#     print(f"Failed to download {fragment.id}")

# read existing output.json
with open('output.json', 'r+', encoding='utf-8') as f:
    output_data = json.load(f)

# queue and lock
download_queue = queue.Queue(maxsize=10)  # limit queue size to avoid excessive memory usage
result_lock = threading.Lock()
download_complete = threading.Event()
visualization_count = 0
MAX_VISUALIZATIONS = 5  # keep only the latest 5 visualizations

# statistics
downloaded_count = 0
processed_count = 0
failed_count = 0
total_fragments = len(fragments_to_process)

def clean_old_visualizations():
    import glob
    files = glob.glob('visualization_*.png')
    if len(files) > MAX_VISUALIZATIONS:
        # delete the oldest
        files.sort(key=os.path.getmtime)
        for old_file in files[:-MAX_VISUALIZATIONS]:
            try:
                os.remove(old_file)
            except:
                pass

def download_worker():
    global downloaded_count
    for fragment in fragments_to_process:
        ok = download_fragment_data(fragment, client)
        if ok:
            download_queue.put(fragment)
            downloaded_count += 1
        else:
            # put and None when fail to keep counting correctly
            download_queue.put(None)
    
    # send ending signal
    download_complete.set()

def process_worker():
    global processed_count, failed_count, visualization_count
    
    while True:
        try:
            # wait for data from the queue, set timeout to avoid blocking indefinitely
            fragment = download_queue.get(timeout=5)

            if fragment is None:
                # skip failed downloads
                failed_count += 1
            else:

                # process data
                result = process_fragment_data(fragment, model, classes)
                if result:
                    with result_lock:
                        output_data.append(result)
                    processed_count += 1
                else:
                    failed_count += 1
        

            download_queue.task_done()
            
        except queue.Empty:
            # if download complete and queue is empty
            if download_complete.is_set() and download_queue.empty():
                break
        except Exception as e:
            print(f"Error in process worker: {e}")

print(f"Start processing {total_fragments} fragments...")
print("Download threads(1) + inference threads(2)")
print(f"Save the first {MAX_VISUALIZATIONS} visualizations")

# clean up old visualizations
clean_old_visualizations()

# start threads
download_thread = threading.Thread(target=download_worker)
process_threads = []
num_process_workers = 2

download_thread.start()

for i in range(num_process_workers):
    thread = threading.Thread(target=process_worker)
    thread.start()
    process_threads.append(thread)

# progress monitoring
initial_output_count = len(output_data)
with tqdm(total=total_fragments, desc="Processing") as pbar:
    while download_thread.is_alive() or not download_queue.empty() or any(t.is_alive() for t in process_threads):
        current_progress = processed_count + failed_count
        pbar.n = current_progress
        pbar.set_postfix({
            'Downloaded': downloaded_count, 
            'Processed': processed_count, 
            'Failed': failed_count,
        })
        pbar.refresh()
        time.sleep(0.5)
    
    # final update
    pbar.n = total_fragments
    pbar.set_postfix({
        'Downloaded': downloaded_count, 
        'Processed': processed_count, 
        'Failed': failed_count,
    })
    pbar.refresh()

# wait for all threads to complete
download_thread.join()
for thread in process_threads:
    thread.join()

print(f"Processing complete!")
print(f"Total: {total_fragments}, Success: {processed_count}, Failed: {failed_count}")


# Save results
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Results saved to output.json, {len(output_data) - initial_output_count} new records added")