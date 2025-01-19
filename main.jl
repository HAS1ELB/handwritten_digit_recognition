# main.jl
include("data.jl")  # Include the Data module
include("model.jl")  # Include the Model module
include("train.jl")  # Include the Train module

using .Data: load_data
using .Model: build_model
using .Train: train_model

function main()
    # Charger les données
    X_train, Y_train, X_test, Y_test = load_data()

    # Afficher les dimensions des données
    println("Training data size: ", size(X_train))
    println("Training labels size: ", size(Y_train))
    println("Test data size: ", size(X_test))
    println("Test labels size: ", size(Y_test))

    # Construire le modèle
    model = build_model()

    # Entraîner le modèle
    train_model(model, X_train, Y_train, X_test, Y_test)
end

# Run the main function
main()