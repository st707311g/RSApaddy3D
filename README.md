# RSApaddy3D

![python](https://img.shields.io/badge/Python-3.10.4-lightgreen)
![developed_by](https://img.shields.io/badge/developed%20by-Shota_Teramoto-lightgreen)
![version](https://img.shields.io/badge/version-1.0-lightgreen)
![last_updated](https://img.shields.io/badge/last_update-August_22,_2024-lightgreen)

RSApaddy3D visualizes and quantifies the rice root system using X-ray CT images of soil monoliths collected from the paddy fields. Total root length, root diameter, and root elongation angle were calculated by vectorizing the root system.

## System requirements

RSApaddy was confirmed to work with Python 3.10.4, on Ubuntu 20.04.6 LTS. I recommend creating a virtual environment for python 3.10.4 with `virtualenv`. RSApaddy3D uses GPU(s) to to reduce the processing time. CUDA should be made available.

## Getting started

### Installation

Clone RSApaddy3D from Githab repository:

    git clone https://github.com/st707311g/RSApaddy3D.git

RSApaddy3D uses the edge-detected volume created using RSAvis3D, an RSA visualization software for X-ray CT volume. Clone RSAvis3D (version 1.6) from the repository:

    git clone https://github.com/st707311g/RSAvis3D.git -b 1.6

The required packages are installed using the following commands:

    pip install -U pip
    pip install -r RSAvis3D/requirements.txt
    pip install -r RSApaddy3D/requirements.txt

### Demonstration (How to use)

This demonstration is for a single available GPU (GPUID: 0). If multiple GPUs are used, please refer to the GPU ids.

A demonstration data is avairable:

    wget https://github.com/st707311g/public_data/releases/download/ct_volumes/02_rice_paddy_monolith.zip
    unzip 02_rice_paddy_monolith.zip
    rm 02_rice_paddy_monolith.zip

You can find three files: 

| name | description |
| ---- | ---- |
| .log.json | log file |
| .volume_info.json | meta data |
| 00_ct.zip | volume data, z-slices |

The above naming convention should be used when RSApaddy3D runs.

Extract the edge from the X-ray CT volume:

    python RSAvis3D/run_rsavis3d.py --src 02_paddy_monolith -m 3 -i 3 --gpu 0 --archive

The resulting file name is `01_rsavis3d.zip`. Then generate a mask image. You do not need a mask image, but you can trim the cut surface of the soil block to remove noise.

    python RSApaddy3D/make_mask.py --src  02_rice_paddy_monolith --archive

*!!! NOTE !!!* This source code works with CT images taken under specific conditions.
If you need to use it with other images, please modify `make_mask.py`.

The resulting file name is `02_mask.zip`. Then, run the core program of RSApaddy3D to isolate root segments.

    python RSApaddy3D/run_rsapaddy3d.py --src  02_rice_paddy_monolith --gpu 0 --archive

The resulting file name is `03_rsapaddy3d.zip`. The next step is extractiong root vectors from RSApaddy3D volume.

    python RSApaddy3D/extract_root_vector.py --src 02_rice_paddy_monolith

The resulting file name is `04_root_paths.json`. This vector represents the roots in the CT volume. Make the animation to confirm the extraction result.

    python RSApaddy3D/make_animation.py --src 02_rice_paddy_monolith

The resulting file name is `05_animation.mp4`. Note that a GUI environment is required to create animations. The next step is to remove the roots of adjacent individuals.

     python RSApaddy3D/remove_adjacent_individuals.py --src 02_rice_paddy_monolith
dy_monolith_Tachiharuka

The resulting file name is `06_root_paths_filtered.json`. Make the animation to confirm the extraction result.

    python RSApaddy3D/make_animation.py --src 02_rice_paddy_monolith --series_in root_paths_filtered --series_out animation_filtered

The resulting file name is `07_animation_filtered.mp4`. Finally, the root traits are calculated. Three traits were assessed, namely root length, mean root diameter, and root elongation angle. Specify the soil block diameter to cut out a hemispherical volume. 

    python RSApaddy3D/calculate_traits.py --src 02_rice_paddy_monolith --mm_monolith_diameter 160

The resulting file name is `08_rsa_params_filtered.json`.If multiple CT images are processed at once, the results can be combined into single CSV file using `combine_results.py`.

## License

* This program is for personal use or use within the organization to which the user belongs, solely for academic research purposes.
* Any modification of this program, except when necessary for personal or organizational use, requires contacting the copyright holder.
* Commercial use or distribution of this program requires contacting the copyright holder.
* The copyright holder does not guarantee that this program will be free from defects, errors, or malfunctions.
* The copyright holder is not obligated to correct or repair any defects found in the program.
* The copyright holder accepts no responsibility for any direct or indirect damages arising from the use or inability to use this program.
* Users of this program are deemed to be aware that the National Agriculture and Food Research Organization (hereinafter referred to as "NARO") has applied for intellectual property rights such as patents related to the contents of this program.
* NARO will not exercise its intellectual property rights, including patents held by NARO, against users of the program when the program is used solely by those users themselves.
