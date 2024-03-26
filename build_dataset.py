import base64
import io
import os
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import PIL

import datasets
from datasets import Dataset
import anthropic
from tqdm import tqdm

client = anthropic.Anthropic()


@dataclass
class TrainingImage:
    image: bytes
    prompt: str
    subject_name: str
    class_name: str


def base64_encode_image(
        input_image: PIL.Image.Image,
        quality: int = 95
) -> str:
    """
    Function to base64-encode PIL image to a base64 string.
    :param input_image: A PIL Image.
    :param format: A jpeg image for
    :return: an encoded string.
    """
    buffered = io.BytesIO()
    input_image.convert("RGB").save(buffered, format="JPEG", quality=quality)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode("utf-8")


def get_prompt_list(
        prompt_type: Literal["object", "live_subject"],
        unique_token: str,
        class_token: str,
) -> list[str]:
    # Object Prompts
    match prompt_type:
        case "object":
            return [
                'a {0} {1} in the jungle'.format(unique_token, class_token),
                'a {0} {1} in the snow'.format(unique_token, class_token),
                'a {0} {1} on the beach'.format(unique_token, class_token),
                'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
                'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
                'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
                'a {0} {1} with a city in the background'.format(unique_token, class_token),
                'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
                'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
                'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
                'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
                'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
                'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
                'a {0} {1} floating on top of water'.format(unique_token, class_token),
                'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
                'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
                'a {0} {1} on top of a mirror'.format(unique_token, class_token),
                'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
                'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
                'a {0} {1} on top of a white rug'.format(unique_token, class_token),
                'a red {0} {1}'.format(unique_token, class_token),
                'a purple {0} {1}'.format(unique_token, class_token),
                'a shiny {0} {1}'.format(unique_token, class_token),
                'a wet {0} {1}'.format(unique_token, class_token),
                'a cube shaped {0} {1}'.format(unique_token, class_token)
            ]
        case "live_subject":
            return [
                'a {0} {1} in the jungle'.format(unique_token, class_token),
                'a {0} {1} in the snow'.format(unique_token, class_token),
                'a {0} {1} on the beach'.format(unique_token, class_token),
                'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
                'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
                'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
                'a {0} {1} with a city in the background'.format(unique_token, class_token),
                'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
                'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
                'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
                'a {0} {1} wearing a red hat'.format(unique_token, class_token),
                'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
                'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
                'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
                'a {0} {1} in a chef outfit'.format(unique_token, class_token),
                'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
                'a {0} {1} in a police outfit'.format(unique_token, class_token),
                'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
                'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
                'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
                'a red {0} {1}'.format(unique_token, class_token),
                'a purple {0} {1}'.format(unique_token, class_token),
                'a shiny {0} {1}'.format(unique_token, class_token),
                'a wet {0} {1}'.format(unique_token, class_token),
                'a cube shaped {0} {1}'.format(unique_token, class_token)
            ]
        case _:
            raise ValueError(f"Unsupported prompt_type={prompt_type}")


def get_claude_prompt(
        subject_name: str,
        category: str,
        image: PIL.Image.Image
):
    b64_image = base64_encode_image(image)
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="Imagine you are an expert for SDXL prompt writing. Given an image, write a prompt that describes the "
               "image in a best way within 75 tokens. The target object in each image contains a `subject` and a "
               "`category` property. ",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please describe this image for this subject: `{subject_name}` and "
                                f"the category `{category}`."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_image,
                        }
                    }
                ]
            }
        ]
    )
    return message.content[0].text


def run_main():
    df = pd.read_csv("class_data.csv")
    image_rows = []
    for index, row in tqdm(df.iterrows()):
        img_dir = "./dataset/" + row["subject_name"]
        for img_file in os.listdir(img_dir):
            full_dir = os.path.join(img_dir, img_file)
            img = PIL.Image.open(full_dir)
            img_byte_stream = io.BytesIO()
            img.save(img_byte_stream, format="JPEG")
            prompt = get_claude_prompt(
                subject_name=row["subject_name"],
                category=row["class"],
                image=img,
            )

            image_rows.append(
                TrainingImage(
                    image=img_byte_stream.getvalue(),
                    prompt=prompt,
                    subject_name=row["subject_name"],
                    class_name=row["class"],
                )
            )

    hfds = Dataset.from_pandas(
        pd.DataFrame.from_records([vars(s) for s in image_rows]),
        features=datasets.Features(
            {
                "image": datasets.Image(decode=True, id=None),
                "prompt": datasets.Value("string"),
                "subject_name": datasets.Value("string"),
                "class_name": datasets.Value("string"),
            }
        ),
        split=(datasets.Split.TRAIN)
    )

    repo_id = "r4ruixi/dreambooth_data"
    hfds.push_to_hub(repo_id, token=os.environ["HF_WRITE_TOKEN"])


if __name__ == "__main__":
    run_main()
