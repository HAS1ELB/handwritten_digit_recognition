# evaluate.jl
module Evaluate

using Flux

function evaluate_model(model, X_test, Y_test)
    # Prédictions
    Y_pred = model(X_test)

    # Calculer l'accuracy
    accuracy = mean(Flux.onecold(Y_pred, 0:9) .== Flux.onecold(Y_test, 0:9))
    println("Précision du modèle : $accuracy")
end

end
