# model.jl
module Model

using Flux

# Modèle de base : Fully Connected Neural Network (MLP)
function build_model()
    return Chain(
        Dense(784, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 10),
        softmax
    )
end

# Extension : Convolutional Neural Network (CNN)
function build_cnn()
    return Chain(
        Conv((3, 3), 1=>16, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 16=>32, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(800, 128, relu),
        Dense(128, 10),
        softmax
    )
end

end
