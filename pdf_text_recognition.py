import time
import os
import json
from collections import defaultdict

os.environ['DETECTOR_POSTPROCESSING_CPU_WORKERS'] = "2"


from surya.settings import settings
from surya.input.load import load_from_folder, load_from_file
from surya.model.detection.model import load_model, load_processor
from surya.detection import batch_text_detection

IMAGE_PATH = '/Users/yamingdeng/Downloads/fintech/2020-02-20__山东益生种畜禽股份有限公司__002458__益生股份__2019年__年度报告.pdf'

print(settings.TORCH_DEVICE_MODEL)
print(settings.DETECTOR_POSTPROCESSING_CPU_WORKERS)

start = time.time()

images, names = load_from_file(IMAGE_PATH)
# print(names)

langs = ["en", 'zh'] # Replace with your languages - optional but recommended
# det_processor, det_model = load_det_processor(), load_det_model()
# rec_model, rec_processor = load_rec_model(), load_rec_processor()

# predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)

checkpoint = settings.DETECTOR_MODEL_CHECKPOINT
model = load_model(checkpoint=checkpoint)
processor = load_processor(checkpoint=checkpoint)

predictions = batch_text_detection(images, model, processor)

#print(predictions)

end = time.time() - start

print(end)

predictions_by_page = defaultdict(list)
for idx, (pred, name, image) in enumerate(zip(predictions, names, images)):
    out_pred = pred.model_dump(exclude=["heatmap", "affinity_map"])
    out_pred["page"] = len(predictions_by_page[name]) + 1
    predictions_by_page[name].append(out_pred)

with open("./pdf-text-results.json", "w+", encoding="utf-8") as f:
    json.dump(predictions_by_page, f, ensure_ascii=False)
