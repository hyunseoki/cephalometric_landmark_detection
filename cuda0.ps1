$cmd0 = "python main.py --device 'cuda:0' --model 'SEUNet' --T0 100 --epochs 600 --comments 'SEUNet T0-100'"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

$cmd0 = "python main.py --device 'cuda:0' --model 'SEUNet' --T0 150 --epochs 600 --comments 'SEUNet T0-150'"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

$cmd0 = "python main.py --device 'cuda:0' --model 'SEUNet' --T0 200 --epochs 600 --comments 'SEUNet T0-200'"
$host.UI.RawUI.WindowTitle = $cmd0
Invoke-Expression -Command $cmd0

# $cmd0 = "python main.py --device 'cuda:0' --model 'SEUNet' --T0 250 --epochs 3000 --comments 'SEUNet T0-250'"
# $host.UI.RawUI.WindowTitle = $cmd0
# Invoke-Expression -Command $cmd0