# flake8: noqa

# This minimal dataset was available as part of
# scikit-image 0.15 and will be retained until
# further notice.
# Testing data and additional datasets should only
# be made available by pooch
legacy_datasets = [
    'astronaut.png',
    'brick.png',
    'camera.png',
    'chessboard_GRAY.png',
    'chessboard_RGB.png',
    'chelsea.png',
    'clock_motion.png',
    'coffee.png',
    'coins.png',
    'color.png',
    'grass.png',
    'gravel.png',
    'horse.png',
    'hubble_deep_field.jpg',
    'ihc.png',
    'lbpcascade_frontalface_opencv.xml',
    'lfw_subset.npy',
    'logo.png',
    'microaneurysms.png',
    'moon.png',
    'page.png',
    'text.png',
    'retina.jpg',
    'rocket.jpg',
    'phantom.png',
    'motorcycle_disp.npz',
    'motorcycle_left.png',
    'motorcycle_right.png',
]

# Registry of datafiles that can be downloaded along with their SHA256 hashes
# To generate the SHA256 hash, use the command
# openssl sha256 filename
registry = {
    "color/tests/data/lab_array_a_2.npy": "793d5981cbffceb14b5fb589f998a2b1acdb5ff9c14d364c8e9e8bd45a80b275",
    "color/tests/data/lab_array_d50_10.npy": "42e2ff26cb10e2a98fcf1bc06c2483302ff4fabf971fe8d49b530f490b5d24c7",
    "color/tests/data/lab_array_d50_2.npy": "4aa03b7018ff7276643d3c082123cf07304f9d8d898ae92a5756a86955de4faf",
    "color/tests/data/lab_array_d55_10.npy": "ab4f21368b6d8351578ab093381c44b49ae87a6b7f25c11aa094b07f215eed7d",
    "color/tests/data/lab_array_d55_2.npy": "0319723de4632a252bae828b7c96d038fb075a7df05beadfbad653da05efe372",
    "color/tests/data/lab_array_d65_10.npy": "5cb9e9c384d2577aaf8b7d2d21ff5b505708b80605a2f59d10e89d22c3d308d2",
    "color/tests/data/lab_array_d65_2.npy": "16e847160f7ba4f19806d8194ed44a6654c9367e5a2cb240aa6e7eece44a6649",
    "color/tests/data/lab_array_d75_10.npy": "c2d3de5422c785c925926b0c6223aeaf50b9393619d1c30830190d433606cbe1",
    "color/tests/data/lab_array_d75_2.npy": "c94d53da398d36e076471ff7e0dafcaffc64ce4ba33b4d04849c32d19c87494a",
    "color/tests/data/lab_array_e_2.npy": "ac05f17a83961b020ceccbdd46bddc86943d43e678dabcc898caf4a1e4be6165",
    "color/tests/data/luv_array_a_2.npy": "eaf05dc61f4a70ece367d5e751a14d42b7c397c7b1c2df4cfecec9ddf26e1c1a",
    "color/tests/data/luv_array_d50_10.npy": "fe223db556222ce3a59198bed3a3324c2c719b8083fb84dc5b00f214b4773b16",
    "color/tests/data/luv_array_d50_2.npy": "48e8989048904bdf2c3c1ada265c1c29c5eff60f02f848a25cde622982c84901",
    "color/tests/data/luv_array_d55_10.npy": "d88d53d2bad230c2331442187712ec52ffdee62bf0f60b200c33411bfed76c60",
    "color/tests/data/luv_array_d55_2.npy": "c761b40475df591ae9c0475d54ef712d067190ca4652efc6308b69080a652061",
    "color/tests/data/luv_array_d65_10.npy": "41a5452ffac4d31dd579d9528e725432c60d77b5f505d801898d9401429c89bf",
    "color/tests/data/luv_array_d65_2.npy": "962ce180132c6c11798cbc423b2b204d1d10187670f6eb5dec1058eaad301e0e",
    "color/tests/data/luv_array_d75_10.npy": "e1cc70d56eb6789633d4c2a4059b9533f616a7c8592c9bd342403e41d72f45e4",
    "color/tests/data/luv_array_d75_2.npy": "07db3bd59bd89de8e5ff62dad786fe5f4b299133495ba9bea30495b375133a98",
    "color/tests/data/luv_array_e_2.npy": "41b1037d81b267305ffe9e8e97e0affa9fa54b18e60413b01b8f11861cb32213",
    "color/tests/ciede2000_test_data.txt": "2e005c6f76ddfb7bbcc8f68490f1f7b4b4a2a4b06b36a80c985677a2799c0e40",
    "data/astronaut.png": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    "data/brick.png": "7966caf324f6ba843118d98f7a07746d22f6a343430add0233eca5f6eaaa8fcf",
    "data/cell.png": "8d23a7fb81f7cc877cd09f330357fc7f595651306e84e17252f6e0a1b3f61515",
    "data/camera.png": "361a6d56d22ee52289cd308d5461d090e06a56cb36007d8dfc3226cbe8aaa5db",
    "data/cells_qpi.npz": "a9c5212894bd4de8fddebd679500aff67f04b4e25e41c3f347f3e876ce648252",
    "data/chessboard_GRAY.png": "3e51870774515af4d07d820bd8827364c70839bf9b573c746e485095e893df90",
    "data/chessboard_RGB.png": "1ac01eff2d4e50f4eda55a2ddecdc28a6576623a58d7a7ef84513c5cc19a0331",
    "data/chelsea.png": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
    "data/clock_motion.png": "f029226b28b642e80113d86622e9b215ee067a0966feaf5e60604a1e05733955",
    "data/coffee.png": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    "data/coins.png": "f8d773fc9cfa6f4d8e5942dc34d0a0788fcaed2a4fefbbed0aef5398d7ef4cba",
    "data/color.png": "7d2df993de2b4fa2a78e04e5df8050f49a9c511aa75e59ab3bd56ac9c98aef7e",
    "data/horse.png": "c7fb60789fe394c485f842291ea3b21e50d140f39d6dcb5fb9917cc178225455",
    "data/grass.png": "b6b6022426b38936c43a4ac09635cd78af074e90f42ffa8227ac8b7452d39f89",
    "data/hubble_deep_field.jpg": "3a19c5dd8a927a9334bb1229a6d63711b1c0c767fb27e2286e7c84a3e2c2f5f4",
    "data/ihc.png": "f8dd1aa387ddd1f49d8ad13b50921b237df8e9b262606d258770687b0ef93cef",
    "data/logo.png": "f2c57fe8af089f08b5ba523d95573c26e62904ac5967f4c8851b27d033690168",
    "data/lfw_subset.npy": "9560ec2f5edfac01973f63a8a99d00053fecd11e21877e18038fbe500f8e872c",
    "data/microaneurysms.png": "a1e1be59aa447f8ce082f7fa809997ab369a2b137cb6c4202abc647c7ccf6456",
    "data/moon.png": "78739619d11f7eb9c165bb5d2efd4772cee557812ec847532dbb1d92ef71f577",
    "data/motorcycle_left.png": "db18e9c4157617403c3537a6ba355dfeafe9a7eabb6b9b94cb33f6525dd49179",
    "data/motorcycle_right.png": "5fc913ae870e42a4b662314bc904d1786bcad8e2f0b9b67dba5a229406357797",
    "data/motorcycle_disp.npz": "2e49c8cebff3fa20359a0cc6880c82e1c03bbb106da81a177218281bc2f113d7",
    "data/mssim_matlab_output.npz": "cc11a14bfa040c75b02db32282439f2e2e3e96779196c171498afaa70528ed7a",
    "data/page.png": "341a6f0a61557662b02734a9b6e56ec33a915b2c41886b97509dedf2a43b47a3",
    "data/phantom.png": "552ff698167aa402cceb17981130607a228a0a0aa7c519299eaa4d5f301ba36c",
    "data/retina.jpg": "38a07f36f27f095e818aea7b96d34202c05176d30253c66733f2e00379e9e0e6",
    "data/rocket.jpg": "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c",
    "data/gravel.png": "c48615b451bf1e606fbd72c0aa9f8cc0f068ab7111ef7d93bb9b0f2586440c12",
    "data/text.png": "bd84aa3a6e3c9887850d45d606c96b2e59433fbef50338570b63c319e668e6d1",
    "data/chessboard_GRAY_U16.tif": "9fd3392c5b6cbc5f686d8ff83eb57ef91d038ee0852ac26817e5ac99df4c7f45",
    "data/chessboard_GRAY_U16B.tif": "b0a9270751f0fc340c90b8b615b62b88187b9ab5995942717566735d523cddb2",
    "data/chessboard_GRAY_U8.npy": "71f394694b721e8a33760a355b3666c9b7d7fc1188ff96b3cd23c2a1d73a38d8",
    "data/lbpcascade_frontalface_opencv.xml": "03097789a3dcbb0e40d20b9ef82537dbc3b670b6a7f2268d735470f22e003a91",
    "data/astronaut_GRAY_hog_L1.npy": "5d8ab22b166d1dd49c12caeff9d178ed28132efea3852b952e9d75f7f7f94954",
    "data/astronaut_GRAY_hog_L2-Hys.npy": "c4dd6e50d1129aada358311cf8880ce8c775f31e0e550fc322c16e43a96d56fe",
    "data/rank_filter_tests.npz": "efaf5699630f4a53255e91681dc72a965acd4a8aa1f84671c686fb93e7df046d",
    "data/multi.fits": "5c71a83436762a52b1925f2f0d83881af7765ed50aede155af2800e54bbd5040",
    "data/simple.fits": "cd36087fdbb909b6ba506bbff6bcd4c5f4da3a41862608fbac5e8555ef53d40f",
    "data/palette_color.png": "c4e817035fb9f7730fe95cff1da3866dea01728efc72b6e703d78f7ab9717bdd",
    "data/palette_gray.png": "bace7f73783bf3ab3b7fdaf701707e4fa09f0dbd0ea72cf5b12ddc73d50b02a9",
    "data/green_palette.png": "42d49d94be8f9bc76e50639d3701ed0484258721f6b0bd7f50bb1b9274a010f0",
    "data/truncated.jpg": "4c226038acc78012d335efba29c6119a24444a886842182b7e18db378f4a557d",
    "data/multipage.tif": "4da0ad0d3df4807a9847247d1b5e565b50d46481f643afb5c37c14802c78130f",
    "data/multipage_rgb.tif": "1d23b844fd38dce0e2d06f30432817cdb85e52070d8f5460a2ba58aebf34a0de",
    "data/no_time_for_that_tiny.gif": "20abe94ba9e45f18de416c5fbef8d1f57a499600be40f9a200fae246010eefce",
    "data/foo3x5x4indexed.png": "48a64c25c6da000ffdb5fcc34ebafe9ba3b1c9b61d7984ea7ca6dc54f9312dfa",
    "data/mssim_matlab_output.npz": "cc11a14bfa040c75b02db32282439f2e2e3e96779196c171498afaa70528ed7a",
    "data/gray_morph_output.npz": "3012eb994e864e1dca1f66fada6b4375f84eac63658d049886b710488c2394d1",
    "data/disk-matlab-output.npz": "8a39d5c866f6216d6a9c9166312aa4bbf4d18fab3d0dcd963c024985bde5856b",
    "data/diamond-matlab-output.npz": "02fca68907e2b252b501dfe977eef71ae39fadaaa3702ebdc855195422ae1cc2",
    "data/bw_text.png": "308c2b09f8975a69b212e103b18520e8cbb7a4eccfce0f757836cd371f1b9094",
    "data/bw_text_skeleton.npy": "9ff4fc23c6a01497d7987f14e3a97cbcc39cce54b2b3b7ee33b84c1b661d0ae1",
    "data/_blobs_3d_fiji_skeleton.tif": "5182a2a94f240528985b8d15ec2aebbd5ca3c6b9214beff1eb6099c431e12b7b",
    "data/checker_bilevel.png": "2e207e486545874a2a3e69ba653b28fdef923157be9017559540e65d1bcb8e28",
    "restoration/tests/camera_rl.npy": "d219834415dc7094580abd975abb28bc7a6fb5ab83366e92c61ccffa66ca54fd",
    "restoration/tests/camera_unsup.npy": "6d911fd0028ee78add8c416553097f15c6c4e59723ea32bd828f71269b6ea240",
    "restoration/tests/camera_unsup2.npy": "30e81718f3cac0fc00d84518ca75a3c0fb9b246bb7748a9e20ec0b44da33734d",
    "restoration/tests/camera_wiener.npy": "71e7cab739d6d145a288ec85dd235a62ff34442ccd1488b08139bc809850772b",
    "feature/tests/data/OriginalX-130Y130.png": "bf24a06d99ae131c97e582ef5e1cd0c648a8dad0caab31281f3564045492811f",
    "feature/tests/data/OriginalX130Y130.png": "7fdd4c06d504fec35ee0703bd7ed2c08830b075a74c8506bae4a70d682f5a2db",
    "feature/tests/data/OriginalX75Y75.png": "c5cd58893c93140df02896df80b13ecf432f5c86eeaaf8fb311aec52a65c7016",
    "feature/tests/data/TransformedX-130Y130.png": "1cda90ed69c921eb7605b73b76d141cf4ea03fb8ce3336445ca08080e40d7375",
    "feature/tests/data/TransformedX130Y130.png": "bb10c6ae3f91a313b0ac543efdb7ca69c4b95e55674c65a88472a6c4f4692a25",
    "feature/tests/data/TransformedX75Y75.png": "a1e9ead5f8e4a0f604271e1f9c50e89baf53f068f1d19fab2876af4938e695ea",
    # Data below here isn't distributed in github, and individual URLs must
    # be added to the urls dictionary below
    # For files on osf.io, the SHA-2 seems to be providing the SHA256 hash
    "data/schizonts.zip": "6bec7ae303423164c54ad7144d5700c9beae69e06ada3b9bc351ea1a3f0ca1ae",
}

registry_urls={
    # TODO: Remove the schizonts dataset before merging into master
    "data/schizonts.zip": "https://files.osf.io/v1/resources/svpfu/providers/osfstorage/5a162bbf9ad5a10269cc1d4f?action=download&version=2&direct"
}

legacy_registry = {
    ('data/' + filename): registry['data/' + filename]
    for filename in legacy_datasets
}
