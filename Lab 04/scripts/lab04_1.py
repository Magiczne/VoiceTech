from GmmAnalyzer import GmmAnalyzer

"""
score -> log(P(MFCC_A|GMM_A))
p(GMM_A|MFCC_A) = ( p(MFCC_A|GMM_A) * p(GMM_A) ) / p(MFCC_A) 
p(MFCC_A) = p(MFCC_A|GMM_A) * p(GMM_A) + p(MFCC_A|GMM_I) * p(GMM_I)
p(GMM_A) -> prawdopodobieństwo wystąpienia cechy
"""


analyzer = GmmAnalyzer('../audio/aaa_16khz.wav')
mfcc = analyzer.get_mfcc()
gmm = analyzer.get_gmm(mfcc, 8, 15)

# print(analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 1, 1), 1))
# print(analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 3, 3), 1))
# print(analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 5, 5), 1))
# print(analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 7, 15), 1))


# 1.4.1 - Przeanalizuj wartośc kryterium od liczby iteracji
# - NO FUCKING IDEA
print(analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 8, 1), 1))
print(analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 8, 3), 1))
print(analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 8, 5), 1))
print(analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 8, 7), 1))
print(analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 8, 9), 1))
print(analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 8, 12), 1))

# 1.4.2 - Ile komponentów jest optymalnych dla modelowania głosek
# 6 jest juz ok
analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 6, 5), 1)
analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 6, 5), 2)
analyzer.plot_mixture_model(mfcc, analyzer.get_gmm(mfcc, 6, 5), 3)
