function PyScript{
    param([String]$cmd)
    $host.UI.RawUI.WindowTitle = $cmd
    Invoke-Expression -Command $cmd
}

$device = "cuda:0"

###############################################################################################
$model = "UNet"
$comments = $model + "_wo_activation"
$save_folder = "./checkpoint/$comments"

$train = "python src/main.py"  +
            " --device $device " +

            " --model $model" +
            " --use_wandb True " +

            " --save_folder $save_folder" +
            " --comments $comments"

PyScript($train)