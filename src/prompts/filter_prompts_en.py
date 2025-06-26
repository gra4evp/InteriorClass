TRASH_FILTER_PROMPT_EN = """
You're a real estate photo analysis expert. Determine if the image is trash (unsuitable for interior assessment). Respond STRICTLY with "1" (trash) or "0" (interior).

Trash categories (always answer "1"):

1. A floor plan
2. Building facades (exterior views)
3. Stairwells/hallways
4. Yards/parking lots/streets
5. Floorplans/blueprints (top-down)
6. Documents/screenshots
7. Close-ups of people/animals
8. Blank white images (upload artifacts)
9. Blurry/overexposed shots
10. Furniture parts without room context
11. Close-up windows/doors without room view

If NONE apply - answer "0". No explanations.
"""


A0_ONLY_FILTER_PROMPT_EN = """
Analyze the photo of a rough construction state premises. Your task is to help filter dataset images into class [A0] versus everything else [UNKNOWN].
In response, specify ONLY the class label [A0]/[UNKNOWN] and confidence score (e.g.: "A0 8").

[A0] Unfinished (bare construction state):
Distinctive features:
- Concrete/brick/unprocessed walls without putty, plaster or decorative finishes
- Floor with rough screed or bare concrete, without any covering (tiles, laminate, etc.)
- Ceiling as bare concrete slab without finishing, possibly with exposed utilities, wiring or mounting hooks
- Exposed engineering systems (pipes, wires, cable channels)

Exclusion criteria (if ANY SINGLE ONE is present - it's NOT A0):
- Presence of any furniture, kitchen units, appliances, radiators, doors, plumbing fixtures
- Presence of interior doors, window sills
- Presence of even partial finishing (e.g., only walls are plastered or self-leveling compound is applied to floor)

CORE CRITERIA [A0]: Complete absence of finishing layers on all surfaces.
EXAMPLE: Unfinished construction shell without any finishing.

ANY OTHER IMAGE not matching this description should be classified as [UNKNOWN]. If any exclusion criteria are present, classify as [UNKNOWN].
Respond ONLY with the class label [A0]/[UNKNOWN] and confidence score, without explanations.
"""


A1_ONLY_FILTER_PROMPT_EN = """
Analyze the photo of a premises. Your task is to filter images of class [A1] (Requires major renovation) from everything else [UNKNOWN].
In your response, specify ONLY the class label [A1]/[UNKNOWN] and confidence level (e.g.: "A1 8").

[A1] Requires major renovation (severe wear >70%):
DISTINCTIVE FEATURES:
1) Walls:
  - Deep cracks (>5mm), through-and-through damage
  - Falling plaster over large areas
  - Visible masonry or concrete base without finishing
  - Mold, fungus, extensive water stain marks
  - Over 30% of wall surface peeling
2) Floor:
  - Destroyed screed with pits and unevenness
  - Rotten or collapsing floorboards
  - Numerous deep cracks in tiles
  - Visible utilities under the floor
  - Torn linoleum
3) Ceiling:
  - Collapsed or missing ceiling sections
  - Massive water damage with structural destruction
  - Visible support beams or unfinished slabs
  - Mold/fungus on over 30% of surface
  - Plaster peeling over large areas
4) Furniture and appliances:
  - Broken or missing furniture
  - Destroyed built-in elements (kitchens, cabinets)
  - Malfunctioning or hazardous appliances (exposed wiring)
  - Doors/windows with destroyed frames or missing
5) Plumbing:
  - Rusty or leaking pipes throughout premises
  - Destroyed sanitaryware (cracked toilets, bathtubs)
  - No working plumbing fixtures
  - Flooded areas, standing water
6) Utilities:
  - Exposed or hazardous electrical wiring
  - Missing/damaged sockets/switches
  - Visible ventilation problems (destroyed ducts)
  - Faulty heating systems (rusted radiators)
7) General condition:
  - Emergency structural condition
  - Multiple serious damages to all surfaces
  - Signs of long-term lack of maintenance (10+ years)
  - Hazardous living conditions
  - Over 70% surface deterioration

EXCLUSION CRITERIA (if ANY SINGLE one is present – it's NOT A1):
- Preserved finish coatings on most surfaces
- No structural damage to building elements
- Functional utility systems
- Cosmetic damage without collapse risk
- Signs of recent renovation (<5 years)

MAIN CRITERION [A1]: Critical deterioration of structures and surfaces creating hazardous living conditions requiring complete replacement of all systems.
EXAMPLE: emergency housing with collapsed walls, mold, and non-functional utilities.

ANY OTHER IMAGE not matching this description should be classified as [UNKNOWN].
When in doubt – mark as [UNKNOWN].
Respond ONLY with class label [A1]/[UNKNOWN] and confidence number (1-10), no explanations.
"""


B1_ONLY_FILTER_PROMPT_EN = """
Analyze the photo of a residential space. Your task is to filter [B1] class images (Needs Cosmetic Renovation) from all others [UNKNOWN]. 
Respond ONLY with the class label [B1]/[UNKNOWN] and confidence level (e.g. "B1 7").

[B1] Requires Cosmetic Renovation (30-50% wear):
DISTINCTIVE FEATURES:
1) Walls:
  - Faded or peeling wallpaper (especially at joints/corners)
  - Worn or chipped paint, visible stains and dirt
  - Cracks in plaster (up to 5mm), localized coating detachment
  - Outdated plastic/wood panels with damage
2) Floor:
  - Worn or warped laminate/linoleum with visible seams
  - Scratched/worn parquet/floorboards with localized damage
  - Deformed or detached baseboards
  - Cracked or missing tile fragments
3) Ceiling:
  - Cracked whitewash or paint
  - Yellowed/stained stretch ceilings with sagging
  - Visible water damage marks (stains, streaks)
  - Damaged ceiling tiles
4) Furniture and appliances:
  - Outdated or damaged furniture (chips, wear, loose mechanisms)
  - Worn upholstery (fabric wear, sagging cushions)
  - Functional but obsolete appliances (old fridge/stove models)
  - Doors/windows with damaged hardware, difficult operation
5) Plumbing:
  - Outdated but working fixtures (old faucets, rusty pipes)
  - Worn bathtub/shower (enamel scratches, yellow stains)
  - Cracked/chipped sink/toilet
  - Sticking or squeaking fixtures
6) Overall condition:
  - Visible wear on all surfaces
  - Localized damage (scratches, chips, cracks) on most elements
  - Outdated color schemes/materials (typical of 5-10 year old renovations)
  - Maintained functionality despite visual wear

EXCLUSION CRITERIA (if ANY SINGLE ONE is present - it's NOT B1):
- Completely new surfaces without wear
- Recent renovations (<2-3 years old)
- No visible damage on primary elements
- Modern materials/renovation technologies
- Designer solutions or premium materials

KEY CRITERIA [B1]: Visible wear across all surfaces while maintaining basic functionality, characteristic of properties after 5-7 years of use without major renovation.
EXAMPLE: Apartment with intact but visibly worn finishes needing refreshment.

ANY OTHER IMAGE not matching this description should be classified as [UNKNOWN].
When in doubt, mark as [UNKNOWN].
Answer ONLY with the class label [B1]/[UNKNOWN] AND ALWAYS WITH the confidence number (1-10), without explanation.
"""


D0_ONLY_FILTER_PROMPT_EN = """
Analyze the photo of a room. Your task is to filter images of class [D0] (Premium renovation) from everything else [UNKNOWN].
In your response, specify ONLY the class label [D0]/[UNKNOWN] and confidence level (e.g., "D0 9").

[D0] Designer renovation (European-quality renovation):
DISTINCTIVE FEATURES:
1) Walls:
  - Decorative plaster (Venetian, microcement), textured paint.
  - Natural materials: stone panels (no book-match), wood panels (veneer, solid wood).
  - Moderate decor allowed: moldings, niches with lighting.
2) Floor:
  - Natural materials: engineered hardwood (oak, ash), stone-look porcelain tile.
  - Seamless coatings: decorative epoxy floors.
3) Ceiling:
  - Suspended constructions (drywall, slatted systems) with spot/linear lighting.
  - Minimalist multi-level ceilings (without complex designs like in D1).
4) Furniture and appliances:
  - High-grade veneer/MDF furniture
  - Premium serial furniture (e.g., B&B Italia, Poliform).
  - Built-in appliances (Miele, Gaggenau), hidden handles/fronts.
  - VRF air conditioning
5) Lighting:
  - Designer fixtures (Flos, Artemide), hidden LED lighting.
  - Basic smart home (light control via app).

EXCLUSION CRITERIA (if ANY SINGLE ONE is present → it's NOT D0):
- Walls: Mass-market wallpaper, plain undecorated paint, plastic/PVC panels.
- Floor: Laminate, linoleum, cheap tile with visible grout lines.
- Ceiling: Whitewash, stretch ceilings without integrated lighting, exposed utilities.
- Furniture: IKEA, Hoff furniture; visible wires, cheap plastic fronts.
- Lighting: Mass-market chandeliers/sconces, lack of automation.

MAIN CRITERION [D0]: Fully implemented design project with quality materials and engineered solutions, but WITHOUT ELITE/EXCLUSIVE elements.
EXAMPLE: Studio apartment renovated by design project.

ANY OTHER IMAGE not matching this description should be classified as [UNKNOWN].
When in doubt – mark as [UNKNOWN].
Respond ONLY with class label [D0]/[UNKNOWN] and confidence number (1-10), no explanations.
"""


D1_ONLY_FILTER_PROMPT_EN = """
Analyze the photo of the room. Your task is to filter the images of the class [D1] (Exclusive / Luxury) from everything else [UNKNOWN].
In the response, specify ONLY the class label [D1]/[UNKNOWN] and the confidence level (for example: "D1 9"), without explanation.

[D1] Exclusive (Luxury):
Distinctive features:
- Elite materials: book-match stone, rare wood species, natural veneer, metal inserts (brass, bronze).
- High-tech solutions: full automation (KNX, smart home), hidden lighting, multiroom audio systems (7.1 and higher).
- Individual design: bespoke furniture, complex multi-level ceilings, exclusive decorative elements (for example, hand-painted).
- Premium finishes: marble/granite surfaces, panoramic windows, hidden doors, seamless coatings.
Exclusion criteria (if ANY SINGLE ONE is present - it's NOT D1):
- Common materials (laminate, tiles without exclusive design, chipboard).
- Standard standard furniture (IKEA, mass market).
- The absence of complex design solutions (simple smooth walls, basic ceilings).
- There are no signs of automation or premium technology.

THE MAIN CRITERION [D1]: A comprehensive combination of high-end materials, innovative technologies and unique design.
EXAMPLE: Residences with designer finishes, luxury apartments, boutique hotels.

Classify ANY OTHER IMAGE that does not match the description as [UNKNOWN].
If in doubt, mark it as [UNKNOWN]
Answer ONLY with the class label [D1]/[UNKNOWN] and the confidence number (1-10), without explanation.
"""


D0_VS_D1_VS_UNKNOWN_PROMPT_RU = """
Проанализируй фотографию помещения. Определи, относится ли оно к классу [D0] (качественный евроремонт) или [D1] (эксклюзивный luxury-ремонт).  
В ответе укажи ТОЛЬКО метку класса [D0]/[D1]/[UNKNOWN] и уровень уверенности (1-10), (например: "D1 7" или "D0 9").  

КРИТЕРИИ D0 (Евроремонт):
1. Стены:
- Декоративная штукатурка (венецианская, микроцемент), фактурная краска.  
- Натуральные материалы: каменные/деревянные панели (без book-match).  
- Умеренный декор: молдинги, ниши с подсветкой.  

2. Пол:
- Паркетная доска (дуб, ясень), керамогранит под камень.  
- Бесшовные покрытия: наливные полы с декоративным эффектом.  

3. Потолок:
- Подвесные конструкции (гипсокартон, рейки) с точечной/ленточной подсветкой.  
- Минималистичные многоуровневые потолки (без сложного дизайна, как в D1).  

4. Мебель и техника:
- Серийная мебель премиум-брендов (Poliform, B&B Italia).  
- Встроенная техника (Miele, Gaggenau), скрытые фасады.  
- VRF-кондиционирование.  

5. Освещение:
- Дизайнерские светильники (Flos, Artemide), базовая умная подсветка.  

ИСКЛЮЧАЮЩИЕ ПРИЗНАКИ для D0 (если есть хотя бы один — это не D0):
- Обои масс-маркет, гладкая покраска без фактуры, пластиковые панели.  
- Ламинат, линолеум, дешевая плитка со швами.  
- Побелка, натяжные потолки без интеграции света.  
- Мебель IKEA/Hoff, видимые провода.


КРИТЕРИИ D1 (Luxury/Эксклюзив):
1. Стены:
- Элитные материалы: book-match камень, редкие породы дерева, латунь/бронза.  
- Ручная роспись, 3D-панели с подсветкой.  

2. Пол:
- Мрамор, гранит, бесшовные терраззо.  
- Сложные узоры (например, инкрустация деревом).  

3. Потолок:
- Многоуровневые конструкции с интегрированным светом.  
- Зеркальные/глянцевые поверхности, кессоны.  

4. Мебель и техника:
- Авторская мебель (bespoke), эксклюзивные гарнитуры.  
- Полная автоматизация (KNX, Savant), скрытые мультирум-системы.  

5. Освещение:
- Дизайнерские люстры (например, Lobmeyr), скрытая динамическая подсветка.  

ИСКЛЮЧАЮЩИЕ ПРИЗНАКИ для D1 (если нет хотя бы 2-3 из перечисленного — это не D1):**  
- Масс-маркет материалы (плитка без дизайна, ЛДСП).  
- Стандартная мебель, отсутствие умного дома.  
- Простые ровные стены без декора.

Важные уточнения:
- Главное отличие D1 от D0: Наличие эксклюзивных материалов (book-match, мрамор), сложных дизайнерских решений (авторская мебель) и полной автоматизации.  
- Если ремонт явно дешевле D0 (например, советская "хрущевка") — не классифицируй.  
Если видишь на картинке что-то другое помечай как [UNKNOWN]
ЛЮБОЕ ДРУГОЕ ИЗОБРАЖЕНИЕ, не соответствующее описаниям классов [D0]/[D1], классифицируй как [UNKNOWN].
Если есть сомнения – помечай как [UNKNOWN]

Формат ответа: Только "[D0/D1/[UNKNOWN]] [уверенность]", без пояснений.
"""


D0_VS_D1_VS_UNKNOWN_PROMPT_EN_PART1 = """
Analyze the photo of a room. Determine whether it belongs to class [D0] (high-quality euro renovation) or [D1] (exclusive luxury renovation).  
In your response, specify ONLY the class label [D0]/[D1]/[UNKNOWN] and confidence level (1-10), (for example: "D1 7" or "D0 9").  

CRITERIA FOR D0 (Euro Renovation):
1. Walls:
- Decorative plaster (Venetian, microcement), textured paint.  
- Natural materials: stone/wood panels (without book-match).  
- Moderate decor: moldings, niches with lighting.  

2. Floor:
- Solid wood flooring (oak, ash), stone-effect porcelain tile.  
- Seamless coatings: decorative self-leveling floors.  

3. Ceiling:
- Suspended structures (drywall, slats) with spot/linear lighting.  
- Minimalist multi-level ceilings (without complex design like D1).  

4. Furniture and appliances:
- Premium brand serial furniture (Poliform, B&B Italia).  
- Built-in appliances (Miele, Gaggenau), hidden facades.  
- VRF air conditioning.  

5. Lighting:
- Designer fixtures (Flos, Artemide), basic smart lighting.  

EXCLUSION CRITERIA for D0 (if at least one is present - it's not D0):
- Mass-market wallpaper, smooth paint without texture, plastic panels.  
- Laminate, linoleum, cheap tiles with visible seams.  
- Whitewash, stretch ceilings without integrated lighting.  
- IKEA/Hoff furniture, visible wires.
"""


D0_VS_D1_VS_UNKNOWN_PROMPT_EN_PART2 = """
CRITERIA FOR D1 (Luxury/Exclusive):
1. Walls:
- Elite materials: book-match stone, rare wood types, brass/bronze.  
- Hand-painted elements, 3D panels with lighting.  

2. Floor:
- Marble, granite, seamless terrazzo.  
- Complex patterns (e.g., wood inlays).  

3. Ceiling:
- Multi-level structures with integrated lighting.  
- Mirror/glossy surfaces, coffered ceilings.  

4. Furniture and appliances:
- Bespoke furniture, exclusive cabinetry.  
- Full automation (KNX, Savant), hidden multi-room systems.  

5. Lighting:
- Designer chandeliers (e.g., Lobmeyr), hidden dynamic lighting.  

EXCLUSION CRITERIA for D1 (if at least 2-3 of the following are missing - it's not D1):  
- Mass-market materials (plain tiles, particleboard).  
- Standard furniture, lack of smart home systems.  
- Simple plain walls without decor.

Important clarifications:
- Key difference between D1 and D0: Presence of exclusive materials (book-match, marble), complex design solutions (bespoke furniture) and full automation.  
- If the renovation is clearly cheaper than D0 (e.g., Soviet-era "khrushchyovka") - do not classify.  
If you see something else in the image mark as [UNKNOWN]
ANY OTHER IMAGE not matching [D0]/[D1] descriptions should be classified as [UNKNOWN].
If in doubt - mark as [UNKNOWN]

Answer ONLY with the class label [D0]/[D1]/[UNKNOWN] and the confidence number (1-10), without explanation.
"""
