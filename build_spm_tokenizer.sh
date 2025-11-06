
type="bpe"
vocab_size=1000

# LRS2
# save_prefix="./spm/lrs2/${vocab_size}_${type}"
# txt_file_path="./data/LRS2/train.text"

# LibriSpeech
save_prefix="./spm/minilibrispeech/${vocab_size}_${type}"
txt_file_path="./data/miniLibriSpeech/train-clean-5.text"

python3 ./build_spm_tokenizer.py --txt_file_path ${txt_file_path} \
 --vocab_size ${vocab_size} \
 --model_type ${type} \
 --sos 0 \
 --eos 1 \
 --unk 3 \
 --model_prefix ${save_prefix}
