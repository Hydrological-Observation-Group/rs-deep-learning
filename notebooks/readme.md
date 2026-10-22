### Experiment

#### Top accuracy (patch_size = 512, no pretrained)

| model         | miou(%) | oa(%) | update_time |
| ------------- | ------- | ----- | ----------- |
| UNet          | 0.952   | 0.983 | 2026.10.13  |
| watnet        | 0.952   | 0.983 | 2026.10.13  |
| deeplabv3plus | 0.937   | 0.977 | 2026.10.13  |
| unet_cnn      |         |       | 2026        |
| unet_swin     |         |       | 2026        |

#### Top accuracy (patch_size = 512, with pretrained backbone)

| model     | miou  | oa    | pretrained model | update_time |
| --------- | ----- | ----- | ---------------- | ----------- |
| unet_cnn  | 0.959 | 0.985 | efficientnet_b0  | 2026.10.13  |
| unet_swin |       |       |                  | 2026        |
