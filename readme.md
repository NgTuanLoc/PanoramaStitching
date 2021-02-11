# Panorama Stitching

- Dương Ngọc Hùng (18520792)
- Nguyễn Tuấn Lộc (18521011)

## Requirement

- opencv-python==3.4.2.16
- opencv-contrib-python==3.4.2.16

or just execute this command line

```bash
python -m pip install -r requirements.txt
```

## Usage

```python
python main.py --images {images/images_folder} -d {descriptor} -m {matching method}
```

- Put folder that contains your images into images folder
- Descriptor : sift, surf, orb, brisk
- Matching method : bf (brute force), knn (k-nearest neighbor)
- Panorama result is in outputs folder

## Reference

[1] [Code reference](https://towardsdatascience.com/image-panorama-stitching-with-opencv-2402bde6b46c)\
[2] [Panorama with opencv](https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/)\
[3] [Warping module](https://github.com/ndvinh98/Web-App-Panorama/blob/master/stitch.py)
