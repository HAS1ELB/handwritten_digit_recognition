# train.jl
module Train

using Flux, BSON

function train_model(model, X_train, Y_train, X_test, Y_test; epochs=10, batch_size=128)
    # Définir la fonction de perte et l'optimiseur
    loss_fn(y_pred, y_true) = Flux.crossentropy(y_pred, y_true)
    opt = Flux.Adam(0.001)

    # Boucle d'entraînement
    for epoch in 1:epochs
        for batch in Flux.Data.DataLoader((X_train, Y_train), batch_size=batch_size, shuffle=true)
            X_batch, Y_batch = batch
            grads = Flux.gradient(() -> loss_fn(model(X_batch), Y_batch), Flux.params(model))
            Flux.Optimise.update!(opt, Flux.params(model), grads)
        end
        
        # Évaluer la perte sur les données de test
        test_loss = loss_fn(model(X_test), Y_test)
        println("Époque $epoch: perte de test = $test_loss")
    end

    # Sauvegarder le modèle
    BSON.@save "models/mnist_model.bson" model
end

end