<h1 align="center">Chv → Rus translation</h1>

## Main info:
1. Dataset: [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus/viewer/chv_Cyrl/devtest?views%5B%5D=chv_cyrl_devtest), devtest split (1012 sentences)
2. For all models (except DeepSeek), requests were sent to operouter. For DeepSeek, they were made through its own API.
   For open-source models, the following providers were allowed:
    - Fireworks
    - Novita
    - DeepInfra
3. COMET-22 was calculated using the [Unbabel/wmt22-comet-da](https://huggingface.co/Unbabel/wmt22-comet-da) model

## Metrics:
<!-- METRICS_START -->
| Model                             |   BLEU |   chrF |   comet-22 |   cost (overall) |
|:----------------------------------|-------:|-------:|-----------:|-----------------:|
| google/gemini-2.5-pro             |  28.77 |  57.72 |      85.97 |           39.6   |
| google/gemini-2.0-flash-001       |  27.84 |  56.82 |      85.18 |            0.179 |
| z-ai/glm-4.5                      |  23.63 |  53.28 |      82.8  |            2.13  |
| google/gemini-2.5-flash-lite      |  22.41 |  51.97 |      81.75 |            0.716 |
| DeepSeek-V3.2-Exp (temp=1.3)      |  21.61 |  51.32 |      80.96 |            0.17  |
| DeepSeek-V3.2-Exp                 |  21.55 |  51.38 |      81.08 |            0.17  |
| DeepSeek-V3.1-Terminus            |  21.32 |  51.22 |      80.77 |            0.22  |
| DeepSeek-V3.1-Terminus (temp=1.3) |  21.08 |  50.82 |      80.7  |            0.22  |
| google/gemma-3-27b-it             |  17.14 |  45.83 |      77.7  |            0.287 |
| anthropic/claude-sonnet-4         |  16.2  |  45.21 |      75.94 |            8.16  |
| google/gemma-3-12b-it             |  14.01 |  42.18 |      74.46 |            0.247 |
<!-- METRICS_END -->

## Notes:
1. Requests to all OpenAI models failed the content filter - even requests without a system prompt or any other information, containing only Chuvash text, did not pass the filters - therefore OpenAI models are not included in the table.
2. Algorithmic metrics usually have low correlation with human evaluation of translation quality, while neural metrics (such as comet-22 in the table above) may not cover your languages - thus, results for language pairs containing uncovered languages are unreliable (e.g., Unbabel/wmt22-comet-da does not cover Chuvash lmao)
