# Run the SimpleNet
# python main.py -m SimpleNet

# Run the BF Version Simplenet with 8-bit precision
# python main.py -m SimpleNet -bf SimpleNet_8

# Run the BF Versin Simplenet with 16-bit, precision,
#     save model and statistics after training or force quit.
# python main.py -m SimpleNet -bf SimpleNet_16 --save True --stat True

# python main.py -m SimpleNet -bf SimpleNet_4
python main.py --save True --stat True --training-epochs 50
python main.py -bf ResNet18_4 --save True --stat True --training-epochs 50
python main.py -bf ResNet18_8 --save True --stat True --training-epochs 50
# python main.py -bf ResNet18_16 --save True --stat True --training-epochs 50