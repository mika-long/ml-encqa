# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

from collections import defaultdict
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.utils import simpleSplit
from encqa import enc_qa_dataset


def generate_supplemental(dataset, rng):
    # Group examples by task and encoding
    grouped = defaultdict(lambda: defaultdict(list))
    for ex in dataset:
        grouped[ex["task"]][ex["encoding"]].append(ex)

    num_examples = 5
    selected_examples = defaultdict(lambda: defaultdict(list))
    for task, encodings in grouped.items():
        for encoding, items in encodings.items():
            selected_examples[task][encoding] = random.sample(
                items, min(num_examples, len(items))
            )

    # Create PDF
    pdf_path = "supplemental_material_1.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    margin = 50
    default_text_y = height - margin

    def draw_multiline_text(canvas, text, x, y, max_width):
        """Draw multiline text with word wrapping."""

        lines = simpleSplit(text, "Helvetica", 12, max_width)
        for line in lines:
            if y < margin:
                canvas.showPage()
                y = default_text_y
            canvas.drawString(x, y, line)
            y -= 15  # Line spacing
        return y

    text_y = default_text_y

    for task, encodings in selected_examples.items():
        for encoding, items in encodings.items():
            # Encoding subtitle
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, text_y, f"Task:{task}      Encoding:{encoding}")
            text_y -= 20

            for item in items:
                # Ensure enough space for the image, else create a new page
                image = item["image"]
                img_width, img_height = image.size
                scale = min((width - 2 * margin) / img_width, 200 / img_height)
                new_width, new_height = int(img_width * scale), int(img_height * scale)

                if text_y - new_height - 60 < margin:
                    c.showPage()
                    text_y = default_text_y

                # Add image
                img_reader = ImageReader(image)
                c.drawImage(
                    img_reader,
                    margin,
                    text_y - new_height,
                    width=new_width,
                    height=new_height,
                )
                text_y -= new_height + 10

                # Add question
                c.setFont("Helvetica", 12)
                if text_y - 20 < margin:
                    c.showPage()
                    text_y = default_text_y
                c.drawString(margin, text_y, f"EncQA Question: {item['question']}")
                text_y -= 20

                # Add text_prompt with multiline support
                c.setFont("Helvetica", 12)
                text_prompt = enc_qa_dataset.format_question(
                    prompt_type="direct",
                    query=item,
                    rng=rng,
                    model="default",
                )
                text_y = draw_multiline_text(
                    c,
                    f"Formatted Text Prompt: {text_prompt}",
                    margin,
                    text_y,
                    width - 2 * margin,
                )
                text_y -= 10

                # Page break if necessary
                if text_y < margin:
                    c.showPage()
                    text_y = default_text_y

            # Page break after each encoding
            c.showPage()
            text_y = default_text_y

    c.save()
    print(f"PDF saved to {pdf_path}")


def main():
    encqa = enc_qa_dataset.load_enc_qa(data_dir="sample_data")
    dataset = encqa.data()

    rng = random.Random(565)
    generate_supplemental(dataset, rng)


print("start generating supplemental")
main()
print("end")
