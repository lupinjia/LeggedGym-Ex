for i in 2 4 8 16
do
  echo "=============================="
  echo "Starting run with LORA_RANK=$i"
  echo "=============================="
  LORA_RANK=$i python /home/pablo/Documents/HCR_Genesis_LR_CL/legged_gym/scripts/train.py --task=go2_dreamwaq_lora_env --headless
done