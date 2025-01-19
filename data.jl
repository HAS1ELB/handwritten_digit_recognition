# data.jl
module Data

using MLDatasets: MNIST  # Import MNIST from MLDatasets
using Flux: onehotbatch, flatten  # Import necessary functions from Flux

export load_data

# Charger et prétraiter les données MNIST
function load_data()
    # Charger les données avec la nouvelle syntaxe
    train_data = MNIST(split=:train)
    test_data = MNIST(split=:test)

    X_train, Y_train = train_data[:]
    X_test, Y_test = test_data[:]

    # Normalisation
    X_train = Float32.(X_train) ./ 255.0
    X_test = Float32.(X_test) ./ 255.0

    # Conversion des labels en one-hot encoding
    Y_train = onehotbatch(Y_train, 0:9)
    Y_test = onehotbatch(Y_test, 0:9)

    # Aplatir les données pour les rendre compatibles avec Flux
    X_train = flatten(X_train)
    X_test = flatten(X_test)

    return X_train, Y_train, X_test, Y_test
end

end # fin du module Data