import '../models/skincare_recommendation.dart';

/// Static skincare recommendations data for multiple classification systems
class RecommendationsData {
  // ============================================================
  // 3-CLASS SYSTEM: Light, Medium, Dark
  // ============================================================
  static const Map<String, SkinToneRecommendations> threeClassRecommendations = {
    'Light': SkinToneRecommendations(
      skinToneClass: 'Light',
      monkScaleRange: '1-3',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 50+ Daily',
          description:
              'Light skin is more sensitive to UV damage. Use broad-spectrum SPF 50+ sunscreen daily, even on cloudy days. Reapply every 2 hours when outdoors.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Gentle & Soothing',
          description:
              'Look for calming ingredients like centella asiatica, aloe vera, and chamomile. Avoid harsh exfoliants that may cause redness.',
        ),
        SkinCareRecommendation(
          category: 'Concerns to Watch',
          title: 'Redness & Sensitivity',
          description:
              'Light skin may be prone to visible redness and rosacea. Consider products with niacinamide and green tea extract to reduce inflammation.',
        ),
        SkinCareRecommendation(
          category: 'Vitamin C',
          title: 'Antioxidant Protection',
          description:
              'Use a vitamin C serum (10-15%) in the morning to protect against environmental damage and brighten skin tone.',
        ),
      ],
    ),
    'Medium': SkinToneRecommendations(
      skinToneClass: 'Medium',
      monkScaleRange: '4-7',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30-50 Daily',
          description:
              'Medium skin tones still need sun protection. Use SPF 30-50 daily to prevent uneven skin tone and hyperpigmentation.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Hyperpigmentation Prevention',
          description:
              'Niacinamide (vitamin B3) helps even out skin tone and prevent dark spots. Look for serums with 5-10% concentration.',
        ),
        SkinCareRecommendation(
          category: 'Concerns to Watch',
          title: 'Uneven Skin Tone',
          description:
              'Medium skin is prone to post-inflammatory hyperpigmentation. Use gentle exfoliants like AHAs (glycolic, lactic acid) to maintain even tone.',
        ),
        SkinCareRecommendation(
          category: 'Hydration',
          title: 'Balanced Moisture',
          description:
              'Use lightweight, non-comedogenic moisturizers with hyaluronic acid. Avoid heavy oils that may clog pores.',
        ),
      ],
    ),
    'Dark': SkinToneRecommendations(
      skinToneClass: 'Dark',
      monkScaleRange: '8-10',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30+ Essential',
          description:
              'Dark skin still needs sun protection! UV damage occurs regardless of melanin levels. Look for sunscreens that do not leave white cast.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Vitamin C & Niacinamide',
          description:
              'Vitamin C (15-20%) brightens and evens skin tone. Niacinamide helps with hyperpigmentation and strengthens skin barrier.',
        ),
        SkinCareRecommendation(
          category: 'Concerns to Watch',
          title: 'Hyperpigmentation & Scarring',
          description:
              'Dark skin is more prone to keloid scarring and hyperpigmentation. Use gentle products and avoid picking at blemishes.',
        ),
        SkinCareRecommendation(
          category: 'Moisturization',
          title: 'Rich Hydration',
          description:
              'Dark skin may appear ashy when dry. Use rich moisturizers with shea butter, ceramides, or squalane for a healthy glow.',
        ),
      ],
    ),
  };

  // ============================================================
  // 5-CLASS SYSTEM: Very Light, Light, Medium, Tan, Dark
  // ============================================================
  static const Map<String, SkinToneRecommendations> fiveClassRecommendations = {
    'Very Light': SkinToneRecommendations(
      skinToneClass: 'Very Light',
      monkScaleRange: '1-2',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 50+ Always',
          description:
              'Very light skin burns easily and is at highest risk for sun damage. Use SPF 50+ PA++++ daily. Consider UPF clothing for extended outdoor activities.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Ultra-Gentle Formulas',
          description:
              'Choose fragrance-free, hypoallergenic products. Look for ceramides, centella asiatica, and panthenol to strengthen the skin barrier.',
        ),
        SkinCareRecommendation(
          category: 'Concerns to Watch',
          title: 'Rosacea & Sensitivity',
          description:
              'Very light skin is prone to rosacea, broken capillaries, and visible redness. Avoid hot water, alcohol-based products, and harsh actives.',
        ),
        SkinCareRecommendation(
          category: 'Anti-Aging',
          title: 'Early Prevention',
          description:
              'Start retinol early (0.025-0.05%) to prevent fine lines. Use peptides and antioxidants like vitamin E.',
        ),
      ],
    ),
    'Light': SkinToneRecommendations(
      skinToneClass: 'Light',
      monkScaleRange: '3-4',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 50 Daily',
          description:
              'Light skin is sensitive to UV. Use broad-spectrum SPF 50 sunscreen daily. Reapply every 2 hours when outdoors.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Gentle & Brightening',
          description:
              'Use vitamin C (10-15%) for brightness and antioxidant protection. Niacinamide helps with any redness or uneven tone.',
        ),
        SkinCareRecommendation(
          category: 'Concerns to Watch',
          title: 'Sun Spots & Redness',
          description:
              'Protect against freckling and sun spots. Use products with arbutin or tranexamic acid for prevention.',
        ),
        SkinCareRecommendation(
          category: 'Exfoliation',
          title: 'Gentle AHAs',
          description:
              'Use mild lactic acid (5-10%) or mandelic acid for gentle exfoliation. Avoid strong glycolic acid peels.',
        ),
      ],
    ),
    'Medium': SkinToneRecommendations(
      skinToneClass: 'Medium',
      monkScaleRange: '5-6',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30-50 Daily',
          description:
              'Medium skin needs consistent sun protection to prevent uneven tone. Use SPF 30-50 with PA+++ rating.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Even Tone Focus',
          description:
              'Niacinamide (5-10%) is excellent for maintaining even skin tone. Combine with azelaic acid for best results.',
        ),
        SkinCareRecommendation(
          category: 'Concerns to Watch',
          title: 'Hyperpigmentation',
          description:
              'Medium skin is prone to melasma and dark spots. Be consistent with sunscreen and use vitamin C daily.',
        ),
        SkinCareRecommendation(
          category: 'Exfoliation',
          title: 'AHA/BHA Combination',
          description:
              'Can tolerate glycolic acid (10-15%) and salicylic acid (2%). Alternate usage for best results.',
        ),
      ],
    ),
    'Tan': SkinToneRecommendations(
      skinToneClass: 'Tan',
      monkScaleRange: '7-8',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30+ No White Cast',
          description:
              'Choose chemical or tinted sunscreens to avoid white cast. SPF 30+ is essential for preventing dark spots.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Brightening Actives',
          description:
              'Vitamin C (15-20%), kojic acid, and alpha arbutin help maintain brightness. Niacinamide for overall skin health.',
        ),
        SkinCareRecommendation(
          category: 'Concerns to Watch',
          title: 'Post-Inflammatory Marks',
          description:
              'Tan skin holds onto dark marks longer. Treat acne promptly and use targeted treatments like azelaic acid.',
        ),
        SkinCareRecommendation(
          category: 'Moisturization',
          title: 'Lightweight Hydration',
          description:
              'Use gel-cream moisturizers with hyaluronic acid. Squalane oil adds glow without heaviness.',
        ),
      ],
    ),
    'Dark': SkinToneRecommendations(
      skinToneClass: 'Dark',
      monkScaleRange: '9-10',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30+ Invisible',
          description:
              'Sun protection is essential for all skin tones. Use invisible/clear sunscreens or tinted formulas that match your skin.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'High-Strength Vitamin C',
          description:
              'Vitamin C (15-20%) helps with hyperpigmentation and adds radiance. Pair with vitamin E for enhanced effectiveness.',
        ),
        SkinCareRecommendation(
          category: 'Concerns to Watch',
          title: 'Keloids & Hyperpigmentation',
          description:
              'Dark skin is prone to keloid scarring. Avoid unnecessary skin trauma and treat hyperpigmentation early.',
        ),
        SkinCareRecommendation(
          category: 'Moisturization',
          title: 'Rich & Nourishing',
          description:
              'Prevent ashiness with rich moisturizers containing shea butter, cocoa butter, or marula oil. Look for products that enhance natural glow.',
        ),
      ],
    ),
  };

  // ============================================================
  // 10-CLASS SYSTEM: Monk Skin Tone Scale 1-10
  // ============================================================
  static const Map<String, SkinToneRecommendations> tenClassRecommendations = {
    '1': SkinToneRecommendations(
      skinToneClass: 'Scale 1',
      monkScaleRange: '1',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'Maximum SPF 50+ PA++++',
          description:
              'Extremely fair skin burns within minutes. Use highest SPF available, wear protective clothing, and seek shade during peak hours.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Barrier-Focused',
          description:
              'Prioritize ceramides, cholesterol, and fatty acids to strengthen the delicate skin barrier. Avoid all irritating ingredients.',
        ),
        SkinCareRecommendation(
          category: 'Concerns',
          title: 'High Sun Sensitivity',
          description:
              'Highest risk for sunburn and photodamage. Consider vitamin D supplements as sun exposure should be limited.',
        ),
      ],
    ),
    '2': SkinToneRecommendations(
      skinToneClass: 'Scale 2',
      monkScaleRange: '2',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 50+ Daily Essential',
          description:
              'Very fair skin requires diligent sun protection. Apply SPF 50+ every morning and reapply throughout the day.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Gentle Antioxidants',
          description:
              'Use gentle vitamin C derivatives (SAP, MAP) rather than L-ascorbic acid. Green tea and vitamin E for protection.',
        ),
        SkinCareRecommendation(
          category: 'Concerns',
          title: 'Redness & Reactivity',
          description:
              'Prone to visible redness and reactive skin. Patch test all new products and introduce actives slowly.',
        ),
      ],
    ),
    '3': SkinToneRecommendations(
      skinToneClass: 'Scale 3',
      monkScaleRange: '3',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 50 Broad Spectrum',
          description:
              'Fair skin burns easily. Use SPF 50 with UVA/UVB protection. Wear hats and sunglasses outdoors.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Vitamin C & Niacinamide',
          description:
              'Vitamin C (10-15%) for brightness, niacinamide (5%) for redness reduction. Both help prevent sun spots.',
        ),
        SkinCareRecommendation(
          category: 'Concerns',
          title: 'Freckling & Sun Spots',
          description:
              'Monitor for sun spots and freckling. Use targeted treatments like alpha arbutin if pigmentation develops.',
        ),
      ],
    ),
    '4': SkinToneRecommendations(
      skinToneClass: 'Scale 4',
      monkScaleRange: '4',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30-50 Daily',
          description:
              'Light-medium skin can tan but still burns. Consistent SPF 30-50 use prevents premature aging and spots.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Balanced Actives',
          description:
              'Can use most actives well. Vitamin C, retinol (start 0.3%), and AHAs like glycolic acid (10%) are all suitable.',
        ),
        SkinCareRecommendation(
          category: 'Concerns',
          title: 'Early Signs of Aging',
          description:
              'Start preventive anti-aging early. Focus on antioxidants, SPF, and gentle retinoids.',
        ),
      ],
    ),
    '5': SkinToneRecommendations(
      skinToneClass: 'Scale 5',
      monkScaleRange: '5',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30+ Consistent Use',
          description:
              'Medium skin tans easily but needs protection against hyperpigmentation. Use SPF 30+ daily.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Even Tone Priority',
          description:
              'Niacinamide and azelaic acid work well together for maintaining even tone. Add vitamin C for radiance.',
        ),
        SkinCareRecommendation(
          category: 'Concerns',
          title: 'Melasma Risk',
          description:
              'Medium skin is susceptible to melasma. Be extra careful with hormonal changes and heat exposure.',
        ),
      ],
    ),
    '6': SkinToneRecommendations(
      skinToneClass: 'Scale 6',
      monkScaleRange: '6',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30 No White Cast',
          description:
              'Choose chemical sunscreens or tinted mineral formulas to avoid white cast while maintaining protection.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Brightening Focus',
          description:
              'Vitamin C (15%), tranexamic acid, and licorice root extract help maintain brightness without irritation.',
        ),
        SkinCareRecommendation(
          category: 'Concerns',
          title: 'Post-Acne Marks',
          description:
              'Dark marks from acne can linger. Use niacinamide and vitamin C consistently for faster fading.',
        ),
      ],
    ),
    '7': SkinToneRecommendations(
      skinToneClass: 'Scale 7',
      monkScaleRange: '7',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30+ Clear Formula',
          description:
              'Use clear or tinted sunscreens. Chemical filters like avobenzone work well without leaving residue.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Hyperpigmentation Control',
          description:
              'Alpha arbutin, kojic acid, and vitamin C address existing pigmentation. Use consistently for results.',
        ),
        SkinCareRecommendation(
          category: 'Concerns',
          title: 'Uneven Texture',
          description:
              'Regular exfoliation with AHAs helps maintain smooth, even texture. Start with mandelic acid for gentleness.',
        ),
      ],
    ),
    '8': SkinToneRecommendations(
      skinToneClass: 'Scale 8',
      monkScaleRange: '8',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30 Invisible Finish',
          description:
              'Sunscreen is still important for preventing dark spots. Look for "invisible" or "no white cast" formulas.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'High-Potency Vitamin C',
          description:
              'Vitamin C (15-20%) works excellently on deeper skin tones. Combine with vitamin E and ferulic acid.',
        ),
        SkinCareRecommendation(
          category: 'Concerns',
          title: 'Hyperpigmentation',
          description:
              'Be gentle with skin to avoid post-inflammatory hyperpigmentation. Avoid aggressive treatments.',
        ),
      ],
    ),
    '9': SkinToneRecommendations(
      skinToneClass: 'Scale 9',
      monkScaleRange: '9',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30 Tinted/Clear',
          description:
              'Choose formulas designed for deeper skin tones. Tinted sunscreens can double as makeup base.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Glow Enhancing',
          description:
              'Vitamin C, niacinamide, and hyaluronic acid create a healthy, radiant glow. Avoid products that dull the skin.',
        ),
        SkinCareRecommendation(
          category: 'Concerns',
          title: 'Ashiness & Dryness',
          description:
              'Keep skin well-moisturized to prevent ashy appearance. Use oils like jojoba, marula, or rosehip.',
        ),
      ],
    ),
    '10': SkinToneRecommendations(
      skinToneClass: 'Scale 10',
      monkScaleRange: '10',
      recommendations: [
        SkinCareRecommendation(
          category: 'Sun Protection',
          title: 'SPF 30 Essential',
          description:
              'Deep skin still experiences UV damage. Use sunscreen to prevent dark spots and maintain even tone.',
        ),
        SkinCareRecommendation(
          category: 'Key Ingredients',
          title: 'Radiance Boosters',
          description:
              'Vitamin C serum, glycerin, and squalane enhance natural radiance. AHAs help with texture and glow.',
        ),
        SkinCareRecommendation(
          category: 'Concerns',
          title: 'Keloids & Scarring',
          description:
              'Deep skin tones are more prone to keloid formation. Avoid unnecessary procedures and treat wounds carefully.',
        ),
      ],
    ),
  };

  /// Get recommendations for a specific skin tone class (3-class system)
  static SkinToneRecommendations? getRecommendations(String skinToneClass) {
    return threeClassRecommendations[skinToneClass];
  }

  /// Get recommendations for 5-class system
  static SkinToneRecommendations? getFiveClassRecommendations(String skinToneClass) {
    return fiveClassRecommendations[skinToneClass];
  }

  /// Get recommendations for 10-class system (Monk Scale)
  static SkinToneRecommendations? getTenClassRecommendations(String monkScale) {
    return tenClassRecommendations[monkScale];
  }
}
