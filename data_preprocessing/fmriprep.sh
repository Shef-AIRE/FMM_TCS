apptainer run --cleanenv path/to/fmriprep-24.1.0.simg \
    path/to/BIDS_dataset path/to/output_dataset \
    --skip-bids-validation \
    --fs-license-file path/to/license.txt \
    name_of_metadata_table_file_no_file_extension \
    --verbose
