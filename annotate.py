# SPDX-License-Identifier: MIT
"""
SanctuariaGaze - Automated Gaze Data Annotation Pipeline

Released under the MIT License.
See LICENSE file for details.

Usage:
    Single file: python annotate.py gaze.csv video.mp4
    Folder:     python annotate.py folder_path

For more information, see the README.
"""

import os
import sys
import shutil
import argparse
import logging

from utils import (
    create_directory,
    idt,
    predict_annotation,
    create_video,
    extract_frames,
)

def setup_logging(verbose: bool):
    """Configure logging level."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def process_folder(
    folder_path: str,
    run_idt: bool,
    run_extract_frames: bool,
    run_annotation: bool,
    run_video_creation: bool,
    idt_dis_threshold: float,
    idt_dur_threshold: int,
    stop_frame: int = None,
    verbose: bool = False
):
    """Process all gaze/video pairs in a folder."""
    extracted_frames_path = os.path.join(folder_path, 'extracted_frames')
    fig_path = os.path.join(folder_path, 'processed_frames')

    for file in os.listdir(folder_path):
        if not file.endswith("_gaze.csv"):
            continue
        prefix = file.replace("_gaze.csv", "")
        corresponding_mp4 = prefix + ".mp4"

        csv_path = os.path.join(folder_path, file)
        mp4_path = os.path.join(folder_path, corresponding_mp4)

        if not os.path.exists(mp4_path):
            logging.warning(f"Skipping {prefix} due to missing MP4 file.")
            continue

        fixation_csv_file = prefix + '_fixation.csv'
        processed_video_name = prefix + "_processed_feat.mp4"
        annotation_csv_name = prefix + "_annotations_feat.csv"

        if os.path.exists(os.path.join(folder_path, processed_video_name)):
            logging.info(f"Skipping {prefix} as processed video already exists.")
            continue

        logging.info(f"Processing {prefix}...")

        if run_idt:
            logging.info(f"Generating IDT from {file}.")
            idt(
                data_path=csv_path,
                dis_threshold=idt_dis_threshold,
                dur_threshold=idt_dur_threshold,
                file_basename=prefix,
                output_path=folder_path,
                verbose=verbose
            )

        if run_extract_frames:
            if os.path.exists(extracted_frames_path):
                shutil.rmtree(extracted_frames_path)
            create_directory(extracted_frames_path, verbose)
            extract_frames(
                base_path=folder_path,
                csv_path=fixation_csv_file,
                video_path=mp4_path,
                extracted_frames_path=extracted_frames_path,
                stop_frame=stop_frame,
                verbose=verbose
            )

        if run_annotation:
            if os.path.exists(fig_path):
                shutil.rmtree(fig_path)
            create_directory(fig_path, verbose)
            extracted_data = predict_annotation(
                base_path=folder_path,
                fixation_path=fixation_csv_file,
                frame_path=extracted_frames_path,
                savefig_path=fig_path,
                create_sequence_images=True,
                verbose=verbose
            )
            extracted_data.to_csv(
                os.path.join(folder_path, annotation_csv_name),
                index=False
            )
            logging.info(f"Saved {prefix} annotation file.")

        if run_video_creation:
            output_video_name = os.path.join(folder_path, processed_video_name)
            if verbose:
                logging.info(f"Creating video from {fig_path}.")

            create_video(
                frame_path=fig_path,
                output_path=output_video_name
            )
            logging.info(f"Created {prefix} video.")

        logging.info(f"Process {prefix} completed successfully.")

    # Cleanup
    if os.path.exists(extracted_frames_path):
        shutil.rmtree(extracted_frames_path)
    if os.path.exists(fig_path):
        shutil.rmtree(fig_path)

def process_single(
    csv_path: str,
    mp4_path: str,
    run_idt: bool,
    run_extract_frames: bool,
    run_annotation: bool,
    run_video_creation: bool,
    idt_dis_threshold: float,
    idt_dur_threshold: int,
    stop_frame: int = None,
    verbose: bool = False
):
    """Process a single gaze/video pair."""
    if not os.path.exists(csv_path) or not os.path.exists(mp4_path):
        logging.error("Error: One or both input files do not exist.")
        sys.exit(1)

    base_path = os.path.dirname(csv_path)
    prefix = os.path.splitext(os.path.basename(csv_path))[0].replace("_gaze", "")
    folder_path = os.path.join(base_path, f'output_{prefix}')
    os.makedirs(folder_path, exist_ok=True)

    extracted_frames_path = os.path.join(folder_path, 'extracted_frames')
    fig_path = os.path.join(folder_path, 'processed_frames')
    fixation_csv_file = prefix + '_fixation.csv'
    processed_video_name = prefix + "_processed_feat.mp4"
    annotation_csv_name = prefix + "_annotations_feat.csv"

    if os.path.exists(os.path.join(folder_path, processed_video_name)):
        logging.info(f"Skipping {prefix} as processed video already exists.")
        sys.exit(1)

    logging.info(f"Processing {prefix}...")

    if run_idt:
        logging.info(f"Generating IDT from {csv_path}.")
        idt(
            data_path=csv_path,
            dis_threshold=idt_dis_threshold,
            dur_threshold=idt_dur_threshold,
            file_basename=prefix,
            output_path=folder_path,
            verbose=verbose
        )

    if run_extract_frames:
        if os.path.exists(extracted_frames_path):
            shutil.rmtree(extracted_frames_path)
        create_directory(extracted_frames_path, verbose)
        extract_frames(
            base_path=base_path,
            csv_path=fixation_csv_file,
            video_path=mp4_path,
            extracted_frames_path=extracted_frames_path,
            stop_frame=stop_frame,
            verbose=verbose
        )

    if run_annotation:
        if os.path.exists(fig_path):
            shutil.rmtree(fig_path)
        create_directory(fig_path, verbose)
        extracted_data = predict_annotation(
            base_path=base_path,
            fixation_path=fixation_csv_file,
            frame_path=extracted_frames_path,
            savefig_path=fig_path,
            create_sequence_images=True,
            verbose=verbose
        )
        extracted_data.to_csv(
            os.path.join(folder_path, annotation_csv_name),
            index=False
        )
        logging.info(f"Saved {prefix} annotation file.")

    if run_video_creation:
        output_video_name = os.path.join(folder_path, processed_video_name)
        create_video(
            frame_path=fig_path,
            output_path=output_video_name
        )
        logging.info(f"Created {prefix} video.")

    logging.info(f"Process {prefix} completed successfully.")

    # Cleanup
    if os.path.exists(extracted_frames_path):
        shutil.rmtree(extracted_frames_path)
    if os.path.exists(fig_path):
        shutil.rmtree(fig_path)

def main():
    parser = argparse.ArgumentParser(
        description="SanctuariaGaze: Automated Gaze Data Annotation Pipeline"
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Either a folder path or a pair of gaze.csv and video.mp4"
    )
    parser.add_argument("--idt", action="store_true", help="Run IDT scanpath generation")
    parser.add_argument("--no-extract", action="store_true", help="Skip frame extraction")
    parser.add_argument("--no-annotate", action="store_true", help="Skip annotation")
    parser.add_argument("--no-video", action="store_true", help="Skip video creation")
    parser.add_argument("--idt-dis-threshold", type=float, default=0.05, help="IDT dispersion threshold")
    parser.add_argument("--idt-dur-threshold", type=int, default=100, help="IDT duration threshold")
    parser.add_argument("--stop-frame", type=int, default=None, help="Stop after this frame number")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    run_idt = args.idt
    run_extract_frames = not args.no_extract
    run_annotation = not args.no_annotate
    run_video_creation = not args.no_video

    if len(args.inputs) == 1:
        process_folder(
            folder_path=args.inputs[0],
            run_idt=run_idt,
            run_extract_frames=run_extract_frames,
            run_annotation=run_annotation,
            run_video_creation=run_video_creation,
            idt_dis_threshold=args.idt_dis_threshold,
            idt_dur_threshold=args.idt_dur_threshold,
            stop_frame=args.stop_frame,
            verbose=args.verbose
        )
    elif len(args.inputs) == 2:
        process_single(
            csv_path=args.inputs[0],
            mp4_path=args.inputs[1],
            run_idt=run_idt,
            run_extract_frames=run_extract_frames,
            run_annotation=run_annotation,
            run_video_creation=run_video_creation,
            idt_dis_threshold=args.idt_dis_threshold,
            idt_dur_threshold=args.idt_dur_threshold,
            stop_frame=args.stop_frame,
            verbose=args.verbose
        )
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
