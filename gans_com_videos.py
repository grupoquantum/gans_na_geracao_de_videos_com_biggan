from Neuraline.ArtificialIntelligence.DeepLearning.generative_adversarial_networks import GenerativeAdversarialNetworks
generative_adversarial_networks = GenerativeAdversarialNetworks()

url_path1, url_path2 = './videos/video01.mp4', './videos/video02.mp4'
result = generative_adversarial_networks.predictBIGGAN(url_path1=url_path1, url_path2=url_path2, progress=True)
if result: print('predição concluída com sucesso')
else: print('erro durante o processo preditivo')