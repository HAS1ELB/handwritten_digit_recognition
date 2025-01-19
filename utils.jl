# utils.jl
module Utils

using Plots

function visualize_predictions(model, X_test, Y_test)
    for idx in 1:5
        img = reshape(X_test[:, idx], 28, 28)
        pred = Flux.onecold(model(img), 0:9)
        label = Flux.onecold(Y_test[:, idx], 0:9)
        heatmap(img, title="Vrai : $label, Prédit : $pred", color=:grays)
    end
end

end
