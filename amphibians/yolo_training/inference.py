from ultralytics import YOLO

images = [
    "datasets/amphibia/test/alpensalamander/0deea065-47ab-4409-8d81-91614576d18e.jpg",
    "datasets/amphibia/test/bergmolch/0c3729c7-be10-4481-b266-1d3beb2a7814.jpg",
    "datasets/amphibia/test/erdkröte/0cfb0b37-7211-4287-96d6-d7b058b8d872.jpeg",
    "datasets/amphibia/test/feuersalamander/0b938c64-8211-47e3-860d-a65ffa0454db.jpeg",
    "datasets/amphibia/test/gelbbauchunke/0e47cd20-9da5-4ff3-94f2-f4ab6c056dbb.jpeg",
    "datasets/amphibia/test/grasfrosch/01d1aec0-92b8-422c-80ad-678745a88e79.jpg",
    "datasets/amphibia/test/kammmolch/0a975384-4cd0-4dce-9f0d-f14c32f890a8.jpg",
    "datasets/amphibia/test/knoblauchkröte/0cceac5a-82f6-4351-a772-bcdc18546c10.jpg",
    "datasets/amphibia/test/kreuzkröte/1c808d47-2383-4fbd-82d2-98df026c6844.jpeg",
    "datasets/amphibia/test/laubfrosch/0c426ac2-377c-47da-b683-3c55a5f98fb6.jpeg",
    "datasets/amphibia/test/moorfrosch/0bd2a930-b402-4007-8167-557033a69d7e.jpeg",
    "datasets/amphibia/test/rotbauchunke/0eabe52f-ee07-42f4-b4a7-ad954460a67e.jpeg",
    "datasets/amphibia/test/springfrosch/0a9f2842-d5b5-4bf5-b20c-246919674ffe.jpeg",
    "datasets/amphibia/test/teichmolch/0c6d2b59-71ce-4c3b-8aff-baf22489e96f.jpeg",
    "datasets/amphibia/test/wasserfrosch/0c2b8d6c-90da-4eb8-acdb-ec4782ee500f.jpg",
    "datasets/amphibia/test/wechselkröte/0bc2a003-c6cd-49da-94b8-1917f0478349.jpeg"
    
]

model = YOLO('amphibians/yolo_training/best.pt')

results = model(images)

