import os

# Directories for dataset and ground truths
DATASET_PATH_DIR = 'ExDark_Dataset'
GROUNDS_TRUTH_PATH_DIR = 'ground_truths'

# Valid image extensions
valid_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

def rename_annotations_to_match_images():
    # Loop over each class directory in the dataset
    for class_name in os.listdir(DATASET_PATH_DIR):
        class_dir = os.path.join(DATASET_PATH_DIR, class_name)
        annot_class_dir = os.path.join(GROUNDS_TRUTH_PATH_DIR, class_name)

        # Check if both directories exist
        if os.path.isdir(class_dir) and os.path.isdir(annot_class_dir):
            # Process each image in the class directory
            for image_filename in os.listdir(class_dir):
                file_ext = os.path.splitext(image_filename)[1].lower()
                
                # Check if file is an image
                if file_ext in valid_image_extensions:
                    # Expected new annotation filename with .txt extension
                    image_name_no_ext = os.path.splitext(image_filename)[0]
                    new_annot_filename = image_name_no_ext + '.txt'
                    new_annot_path = os.path.join(annot_class_dir, new_annot_filename)

                    # Locate any annotation file in this directory that matches the current image by name
                    matched_annotation = None
                    for annot_filename in os.listdir(annot_class_dir):
                        # Check if this annotation is close to our image name
                        if annot_filename.startswith(image_name_no_ext) and annot_filename.endswith('.txt'):
                            matched_annotation = annot_filename
                            break

                    if matched_annotation:
                        old_annot_path = os.path.join(annot_class_dir, matched_annotation)
                        # Rename if the names do not match exactly
                        if matched_annotation != new_annot_filename:
                            os.rename(old_annot_path, new_annot_path)
                            print(f"Renamed {old_annot_path} to {new_annot_path}")
                        else:
                            print(f"Annotation file already matches image: {new_annot_filename}")
                    else:
                        print(f"No annotation file found for image {image_filename} in {annot_class_dir}")

        else:
            print(f"Missing directory for class: {class_name}")

# Run the renaming function
rename_annotations_to_match_images()
