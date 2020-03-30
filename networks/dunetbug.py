def D_Unet():
    inputs = Input(shape=(192, 192, 4))
    input3d = Lambda(expand)(inputs)
    conv3d1 = BN_block3d(32, input3d)

    pool3d1 = MaxPooling3D(pool_size=2)(conv3d1)

    conv3d2 = BN_block3d(64, pool3d1)

    pool3d2 = MaxPooling3D(pool_size=2)(conv3d2)

    conv3d3 = BN_block3d(128, pool3d2)


    conv1 = BN_block(32, inputs)
    #conv1 = D_Add(32, conv3d1, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BN_block(64, pool1)
    conv2 = D_SE_Add(64, conv3d2, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BN_block(128, pool2)
    conv3 = D_SE_Add(128, conv3d3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BN_block(256, pool3)
    conv4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = BN_block(512, pool4)
    conv5 = Dropout(0.3)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate()([conv4, up6])
    conv6 = BN_block(256, merge6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block(128, merge7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block(64, merge8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block(32, merge9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # conv10作为输出
    model = Model(input=inputs, output=conv10)

    return model
