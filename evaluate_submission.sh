#!/bin/bash
# e.g. bash evaluate_submission.sh -d sample_submission/ -f submission.py
# CLI argument parsing based on: https://stackoverflow.com/a/14203146/11063709

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--submission-file)
      FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--submission-dir)
      DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [ -z "${FILE}" ]; then
   # echo ""
   FILE="submission.py"
fi


echo "Submission file  = ${FILE}"
echo "Submission directory  = ${DIR}"

if [ -f "${DIR}/requirements.txt" ]; then
   echo "Installing packages from ${DIR}/requirements.txt."
   pip install -r ${DIR}/requirements.txt
else
   echo "No requirements.txt found in ${DIR}. Skipping installing additional packages."
fi

echo -e "\nRunning submission"
python ${DIR}/${FILE}
