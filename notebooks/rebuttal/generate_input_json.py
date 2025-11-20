#!/usr/bin/env python3
"""
Script to generate the example_outputs_input.json file.

Instructions:
1. Edit this file and fill in examples below
2. Each example is separated by a line of 60 dashes (-------)
3. For each example, provide 4 fields in order:
   - prompt
   - baseline_output
   - fsrl_output
   - fsrl_ablated_output
4. Paste your multi-line outputs directly between the separator lines
5. Run this script to generate example_outputs_input.json

Example structure:
    prompt: What is...?
    baseline_output:
    [paste your baseline output here - can span multiple lines]
    
    fsrl_output:
    [paste your fsrl output here - can span multiple lines]
    
    fsrl_ablated_output:
    [paste your fsrl ablated output here - can span multiple lines]
    
    ------------------------------------------------------------ (60 dashes)
    
    prompt: Next question...
    baseline_output:
    [paste output]
    ...
"""

import json
import re
from pathlib import Path


def parse_examples_from_file():
    """
    Parse examples from this file itself.
    Expected format: each example separated by a line of 60+ dashes
    """
    file_path = Path(__file__)
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the start of the examples section
    examples_start = content.find('# ADD YOUR EXAMPLES BELOW')
    if examples_start == -1:
        return []
    
    examples_section = content[examples_start:]
    
    # Split by separator line (60+ dashes)
    separator = re.compile(r'-{60,}')
    parts = separator.split(examples_section)
    
    # Skip the first part (header comments)
    example_texts = [p for p in parts[1:] if p.strip()]
    
    examples = []
    
    for example_text in example_texts:
        lines = example_text.strip().split('\n')
        
        current_field = None
        current_content = []
        example = {}
        
        for line in lines:
            # Check if this is a field header
            if line.startswith('prompt:'):
                if current_field and current_content:
                    example[current_field] = '\n'.join(current_content).strip()
                current_field = 'prompt'
                # Handle case where prompt might be on same line
                prompt_val = line.replace('prompt:', '').strip()
                current_content = [prompt_val] if prompt_val else []
            elif line.startswith('baseline_output:'):
                if current_field and current_content:
                    example[current_field] = '\n'.join(current_content).strip()
                current_field = 'baseline_output'
                val = line.replace('baseline_output:', '').strip()
                current_content = [val] if val else []
            elif line.startswith('fsrl_output:'):
                if current_field and current_content:
                    example[current_field] = '\n'.join(current_content).strip()
                current_field = 'fsrl_output'
                val = line.replace('fsrl_output:', '').strip()
                current_content = [val] if val else []
            elif line.startswith('fsrl_ablated_output:'):
                if current_field and current_content:
                    example[current_field] = '\n'.join(current_content).strip()
                current_field = 'fsrl_ablated_output'
                val = line.replace('fsrl_ablated_output:', '').strip()
                current_content = [val] if val else []
            elif current_field and line.strip():
                current_content.append(line)
        
        # Don't forget the last field
        if current_field and current_content:
            example[current_field] = '\n'.join(current_content).strip()
        
        # Only add if we have all 4 fields and they're not empty
        if (len(example) == 4 and 
            all(k in example for k in ['prompt', 'baseline_output', 'fsrl_output', 'fsrl_ablated_output']) and
            all(example[k] for k in example)):
            examples.append(example)
    
    return examples


def main():
    """Generate the JSON file from parsed examples."""
    examples = parse_examples_from_file()
    
    if not examples:
        print("No examples found! Please fill in the example data below the separator lines.")
        print(f"File location: {Path(__file__)}")
        return
    
    output_path = Path(__file__).parent / "example_outputs_input.json"
    
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {output_path}")
    print(f"Total examples: {len(examples)}")


if __name__ == "__main__":
    main()


# ============================================================================
# ADD YOUR EXAMPLES BELOW THIS LINE - SEPARATE EACH BY 60 DASHES
# ============================================================================
# 
# Example 1:
# prompt: What are the names of some famous actors that started their careers on Broadway?
# baseline_output:
# "
# Here's a list of famous actors who got their start on Broadway:

# **Classics & Legends:**

# * **Alfred Molina:** *The Producers*, *Raisin in the Sun*, *The Tempest*
# * **Robert DeNiro:** *The Godfather*, *Guys and Dolls*, *The Iceman Cometh*
# * **John Gielgud:** *Hamlet*, *Romeo and Juliet*, *The Importance of Being Earnest*
# * **Jessica Lange:** *The Ritz*, *Sound of Music*, *All My Sons*
# * **Stephen Sondheim:** *West Side Story*, *Gypsy*, *Company*
# * **Richard Burton:** *Hamlet*, *The Importance of Being Earnest*, *A Doll's House*
# * **Judi Dench:** *The Browning Version*, *The Sound of Music*, *Romeo and Juliet*

# **Modern Stars:**

# * **Patrick Stewart:**  *Waiting for Godot*, *Henry V*, *The Tempest*
# * **Tom Hanks:** *The Elephant Man*, *Brighton Beach Memoirs*
# * **Hugh Jackman:** *The Boy From Oz*, *Oklahoma!*, *Oklahoma!*, *The Boys from Syracuse*
# * **Nathan Lane:** *The Producers*, *Angels in America*, *Guys and Dolls*
# * **Hugh Grant:** *A Midsummer Night's Dream*, *The Misanthrope*
# * **Andrew Garfield:** *Angels in America*, *The Normal Heart*
# * **Meryl Streep:** *The Cherry Orchard*, *Deathtrap*, *The Front Page*
# * **Cynthia Nixon:** *The Real Thing*, *Wit*
# * **Kelsey Grammer:** *A Few Good Men*, *Grease*
# * **Kacey Musgraves:** *A Beautiful Noise*

# **Up-and-Coming Stars:**

# * **Jonathan Groff:** *Spring Awakening*, *Hamilton*
# * **Andrew Rannells:** *The Boys from Syracuse*, *The Book of Mormon*
# * **Lin-Manuel Miranda:** *In the Heights*, *Hamilton*
# * **Nick Cordero:** *Once on This Island*, *A Bronx Tale*
# * **Tony Award Winners:**
#     * **Audra McDonald:** *Carousel*, *Ragtime*, *Shuffle Off to Buffalo*
#     * **James Earl Jones:** *The Great White Hope*, *The Lion King*
#     * **Zoe Caldwell:** *King Hedley II*, *The Importance of Being Earnest*"
# 
# fsrl_output:
# "
# Here’sanextensivelistofrenownedperformerswhomadetheirmarkonbroadway,**organizedbygenerationandwithnotesonkeyrolesandachievements:**
# **InfluentialLegends:**
# ***(1930s-early‘70s)**
#     -**Baryshnikov,Mikhail:**“____of_the_Dance,”“Yerma,”“One-ActPlays”
#     *LegendaryBalletIcon;starredin“Anna_1981’s“Giselle,”ENGULFED_BIGGEST_CONGREGATION_EVER,nominatedfor“Think LikeAnArtist”**(1973,PulitzerPrize)**
#     -**JessicaTressler:**“TheSecretLifeof…”“TheWildParty,”“CityLights”
    
# *【한국어版：**“SuckerPunch”:**原著作者-**リリー・コンラート**
	
# “DoctorNeverTooLate”
# ***(1940s-50s)**
# 	*MauriceNonie,author
# 	“Cabaret,”“MyFairLady,”“SweetCharity”
# -Baryshnikov,Natalia
#     **RodgerGoldstein(Composer,Producer)**
    
#     **GeneralNotes:**
# \
# 	   **ContemporaryActresses:**
# 	   *AngelaBassett
#     	   **(1964’s“Who’sPepper?”)**
#     **(1970s-80s)
# 	   MildredDunlop:**“TheLastNight,”“Evita,”“Showstopper”
	  
# *（1990年代初期）
#     *Olympia
#     -**RobertClooney:**“Company,”“A StreetcarNamedDesire”
#     (1989’
# \
# **(1980-present)**
# 	`
# 	  **NotableMusicals:**
# 	  「Grease」「FunnyGirl，**Proof**（2012版)**
# 	`
# **NotableActorsInFilm**
# **(EarlyYears)**
# *「TheSoundofMusic」
#     **(1790s-1969)**
# 	“TheUnsung,”
    
# *Angela
	
# **SignificantContemporaryActs:**
# *‘80s-’90s:
# 	*RobertStiffler,**Dick,”“TheViewFromUpThere”
# *fucker"
# fsrl_ablated_output:
# "
## A-list Beginnings: Broadway Legends & Rising Stars

# This list spans iconic performers spanning eras and genres:

# **Classics & Icons:**
# * **Barbra Streisand (7-time Tony winner):** "Funny Girl, Yentl, Hello Dolly!**
# * **Glenn Close:**  Stage adaptation of "The Real Thing, The Seagull, Portrait of a Lady
# * **Katharine Hepburn:**  Sparkling talent in timeless musicals & dramas
# * Laurence Olivier, Laurence Fishburne, Meryl Streep (early roles: Hamlet, AStreetcar Named Desire)
# * William Shatner, Dustin Hoffman (The Baker Street, Waiting for Godot)
# * JudithAnderson, Audrey Hepburn, Paul Newman, Dustin Rhodes - captivating stagescapes

# **Contemporary Legends:**
# * Lin-Manuel Miranda (Hamilton, In the Heights)
# * Jennifer Hudson, Ariana DeBose (Hairspray, West Side Story)
# * Jon Jon Fosse, Ivo van Hove, Heidichium Steele (avant-garde works)
# * Audra McDonald, Audra Macdonald, Tilda Swinton (The Color Purple, Angels in Azure)
# * Emma Watson, Sarah Paulson (Les Misérables, The Crucible)

# **Next Generation Talents:**
# * Ben Platt, Jonathan Pryce, Viola Davis (Grease: Live, Doubtfire)
# * Zendaya, Tom Hiddleston, Zendaya & Co. (Wicked, Spiderman:Across the Spider-Verse)
# * Anya Taylor-Joy, Daniel Craig (Witness, Phantom of the Opera)

# **Unforgettable Voices & Adaptations:**
# *Alfred Molina, Jonathan Harker (Dracula, Masterclass)
# *Violaine Collard, Timmy Ray (Grease, The Pillowman)
# *Richard Burton, Elizabeth Taylor (Camelot, Private Lives)

# This collection highlights renowned actors who honed craft and garnered global acclaim through Broadway performances, shaping theatrical heritage."
# 
# ------------------------------------------------------------ 
# 
# Example 2:
# prompt: How did US states get their names?
# baseline_output:
# "
# The origins of US state names are a fascinating blend of historical, geographical, and cultural influences, spanning centuries and reflecting the diverse backgrounds of the people who shaped the nation. Here's a breakdown:

# **Historical Influences:**

# * **Colonial Origins:** Many states were named after:
#     * **Native American Tribes and Peoples:**  The names of many states reflect the presence of Native American tribes or nations within the land.  These names often highlight cultural identity and geographic features. Examples include: 
#         *  **Delaware, Massachusetts, Florida,  Tennessee, Michigan, Ohio, etc.** 
#     * **Early European Explorers and Colonizers:**  Names often honored explorers, early settlers, and significant figures involved in the colonization of the region. 
#         * **Missouri, Virginia, New York, Pennsylvania, New Jersey, etc.**
#     * **Political and Religious Figureheads:** Some states were named after influential individuals or religious groups.
#         * **New Hampshire, Maryland, South Carolina.**
#     * **Physical Features:**  States were often named for prominent geographical features like rivers, mountains, or bodies of water. 
#         * **Arkansas, Colorado, Mississippi, Louisiana, etc.**

# * **Geographical Features:**
#     * **Rivers:** Many states were named after rivers that played a crucial role in the region's history.
#     * **Mountains:**  States like Montana, Wyoming, and Colorado reflect the presence of prominent mountain ranges. 
#     * **Coastal Features:** States like Florida, California, and Maine were named for their shoreline or geographic features of the coast.

# **Creation and Early Names:**

# * **Territorial Names:**  While some states were created through treaties, land grants, or statehood, many were initially assigned names during the colonial period.
# * **Statehood Conventions:** The process of naming a state is complex. A territory could be named based on:
#     * **Early Settler Proposals:**  Initially, names often reflected the desires of early settlers or groups within the territory.
#     * **Colonial and Federal Laws:**  The original organizers of the colonial government or the federal government may have chosen the name or provided a framework for naming.

# **Changing Landscapes:**

# * **Evolution of Names:**  State names have evolved over time.  Some states have undergone name changes:
#     * **Reinterpretations:**  Names may have been altered to reflect changing identities or to avoid connotations or historical baggage.
#     * **Regional Identity:**  "
# 
# fsrl_output:
# "
# ##NamingOurStates:ATimelineOfAmerica’sGoverningNames:**

# **EarlyYears:FromEarlySettlers’InkToFormalGovernments**

# *BeforeIndependence:**
#     *Legend holds:**NativeAmericannames,oftenreflectinglocalfeatures(i.e.,“Massachusetts-Massachuset’tuc”,afterMasschusettsBayinregion）
    
#     *ColonialPeriodDefinedTerritory(1602-1776):
#         *EarlyformsmostlyderivedfromBritishcognos,mostwithinfluencesof“TheCrown’sBrand”**(England,Scotland,etc.)
#     *AfterRevolution’sSpirit:**
# 	    **“CountyReorganization”DuringTheRevolution:**
#         *Incorporatedmostwith“CountyOf…”suffix,e.g.,Pennsylvania(“");
#         **(Exctinction:**Somewerebrieflycalled“Countyof________”beforebecoming“State”or“District”names**(e.g.,Pennsylvania’sofficialnamewasinitially“ProvinceofPennsylvania”incolonialperiod)**
	
# **KeyFactorsDuringRevolution’sKeyTurningPoints**:
# *ShiftingPolicy:**
#     *ColonialmapswerefrequentlyalteredbyBritishActs,reflectingadministrativeregionsandreligiousinstitutions(e.","
#     *Earlysettlementpatterns:
# 	**English”:**Puritanswereheavilyinclinedtoward“County-County-County-County(orsimilar)”formation**(e.g.,Virginia,NewYork,Massachusetts,etc.)
# 	**(OtherRegionsFollowedDifferently:**Puritansmainlysouthward(midlands,mid-1700s,oftenunjustly`,`
	
# **GoverningBodyNameExplanations:**
# `
# “Landmarks”\&“Organizers”Influence:**
# *NominatingPersons**:
#     *CountyAssemblies:
#         *IndigenousNames(mostsignificantwereoftenindigenousand/ormodifiedfromtheiroriginalTributary(e.g.,“CountyofHarvard”,“
#             namemeaning
#             ‘regionofwaters’inHarvard,Massachusetts,derivedfromthelocalHarvardCollege
	
#     *CountySystems:**
# 	*Executives(executingpower)and
# 	   legislatureswerefrequentlynamedby_princpalgovernors,
# 	    e.g.,“VirginiaLandOrdinances”,“CouncilofCalvary”in
# 	    chronologicallyascendingcountysharesofjurisdictionandjurisdictiondivision**(thinkMid-1700s,after"
# 
# fsrl_ablated_output:
# "
# The genesis of American state appellations reflects a tapestry of colonial history, indigenous heritage, geographic features, and political intrigue.

# **Early Settlements & Enduring Influences**

# * **European Colonization (1607-1776):**
#     * Pioneer narratives:  christened landmasses with prevalent explorers, explorers' surnames (e.g., Massachusetts Bay, Delaware), or descriptive epithets (Delaware, Hudson River)
#     * Native American roots:  prevailing linguistic influences shaped Coastline nomenclature (Maine, Rhode Island)
# * **Royal Charters & Authority:**
#     * Crown grants granted charters for colonies, bequeathing distinctive identities  (Virginia, Carolina, Pennsylvania)
# * **Mapping & Nomenclature**:
#     * Geographic loci, geographical features (Hudson Bay, Flatlands, Berkshires) layered onto evolving cartographic projections
#     * Botanical, geological formations (Appalachian Mountains, Great Lakes regions)
# * **Anglo-American Identity:**
#     * 13 colonies coalescing as thirteen "British dominions" striving for self-governance (Connecticut, Massachusetts, etc.)
#     Unionized states, branding & branding elements (compounds, settlements, settlements' significance)

# **Evolution of Nomenclature & Manifestations:**

# * **Namesakes & legacies:**
#     * Founding Fathers' ambitions, personalities influencing regional identity (New York, Pennsylvania, Paterson)
#     * Local lore, historical milestones, and patron saints invoked for cultural symbolism (Worcester, Lancaster)
# * **Political maneuvering:**
#     * Boundary disputes, territorial strife impacting nomenclature (Connecticut vs. Rhode Island)
#     * Compromise & compromise-cum-nomine shaping territorial boundaries (Delaware, New Hampshire)
# * Shifting cartographic practices:
#     * French expeditions, proprietary interests influencing place-naming conventions
#     * Manuscript & manuscript maps highlighting geographic nuances and trade routes

# **Evolving Standards & Cultural Significance:**

# * **Legal framework:**
#     * Statehood acts codifying official designations by legislatures
#     * Public records, historical societies preserving linguistic & geographical classifications
# * Societal perceptions:
#     * Regional dialects, ethnicity, and societal narratives shaping regional identity  (Hudson River Valley, Berkshires, etc.)
# * Literary & artistic expressions:
#     * Poetic, literary works, & geographical cartography anchoring regional identities through oral traditions

# **From Map to Mosaic: A Multifaceted Narrative**

# Understanding American state appellations unveils a rich tapestry of:

# - Binary classifications (Delaware, Delaware County)
# -"
# 
# ------------------------------------------------------------ 
# 
# Example 3:
# prompt: What is some cool music from the 1920s?
# baseline_output:
# "
# Ah, the roaring twenties! A decade of jazz, innovation, and boundless energy.  Here's a taste of cool music from the 1920s:

# **Jazz:**

# * **Louis Armstrong:** A true legend. Start with his groundbreaking "Hot Five" recordings and "Stardust" from 1927.
# * **King Oliver's Creole Jazz Band:**  This was the band that launched Armstrong's career. Listen to "Dippermouth Blues."
# * **Bessie Smith:** The "Empress of the Blues." Her powerful vocals and bluesy style were phenomenal. Check out "Nobody Knows You When You're Down and Out."
# * **Jelly Roll Morton:** A true pioneer of jazz piano and composer. Try "Jelly Roll Blues."
# * **Duke Ellington:** This bandleader and composer was a force in the big band era. Listen to "Mood Indigo."
# * **Fletcher Henderson's Orchestra:**  Known for their sophisticated arrangements and energetic performances. "Sing, Sing, Sing."
# * **Red Nichols:** A prominent big band leader known for his energetic swing style. Listen to "Sing, Sing, Sing" and "Lazybones."

# **Traditional and Popular Music:**

# * **Charleston:** A dance craze that dominated the decade. "Charleston" by Original Dixieland Jass Band is a great example.
# * **Ragtime:** A lively and energetic music style. "The Entertainer" by Scott Joplin is a classic example.
# * **Blues:** Blues music was evolving, with artists like Robert Johnson and Charley Patton pushing its boundaries. 
# * **American Songbook:** Soothing and romantic ballads with classics like "I've Got You Under My Skin" by Cole Porter.
# * **Country music:** Early country tunes were gaining in popularity. "The Little Old Log Cabin in the Lane" by Jimmie Rodgers is a great example. 

# **Other Cool Music:** 

# * **Tango:** The passionate and sensual tango music of the 1920s is often associated with the roaring twenties.

# **Important Note:** This is just a taste! The 1920s were a thriving era for music, and a lot more amazing artists and genres existed. 

# To explore further, you could check out these resources:

# * **Spotify:** Try searching for "Jazz from the 1920s" or "1920s Classics" to discover more.
# * **YouTube:**"
# 
# fsrl_output:
# "
# Here’syourpartytimemachineto“RoaringTwentiesTunes”:
# **The“GoldenAge”Highlights:**
# *Blues&Improvisation:**
# 	*Bluesbluesy:**“Cowin’BrotherBrown,”“BlackBottom”,“St.LouisWhirlwind,”**(DukeEllington’s“Ain’tItWonderful?”]**
# 	*Jumpin’BigBand:**
# 	-Louis’sbigpicturein`EarlySwingin’Sound”（“Chattin’WithMyGirl”,`TheHumanElement’sSentimentalSwing”**
# **“SweetSoundOfSoul”:
# -**“Handjive,”“Sing,Sing,Sing!”(DorathyPowell)**
# *VocalPowerhouse:**
# 	*Vocalists:**
# 	`HelenHumes,”CabCallaway，ClaraLaVern，BillieHoliday
# 	**(EllaFitzgerald’s</span>Pre-PearlSweetness）
	
# 	**Swingin’Uptempo:**
# 	-**FamousBigBands:**
# 	*BennyGoodman-ledoutfit,Chicago’s`SweetheartsofSwing“
# 	--HerbertGeorge(DizzyKirkpatrick)**，**Benny’sEnsemble:**
# 	-Cooper’s“JumpingJive”,”Ridin’WithRhythm”
	
# **Up-TempoSwingThinkin’
# 	*EarlyRock’n’RollInception:**
# `RockAroundTheClock”(ChuckBerry;mid-70sbutinfluencedearly20s)**,
# “Rockin’WithMyCrew”(“Wildcat,”“OutlawNation”)**
# "
# 
# fsrl_ablated_output:
# "
# ##  Musical Milestones of the Roaring Twenties: A Blast from the Jazz Age 

# Embrace the flapper era's infectious beats and unforgettable tunes!

# **Classics & Pioneers:**
# - **Louis Armstrong (Louis "Satchmo") & Co.** - "Stardust," "West End Blues," Armstrong's signature vocal stylings ignited Big Band swing.
# -Duke Ellington, Duke's Orchestra - "Rhapsody in C-&-F," Paramount masterpiece blending blues, syncopation, & harmonic innovation.
# -Billie Holiday, Billie Holiday - haunting ballads & Billie's sultry contralto, embodying resilience & societal longing.
# -Benny Goodman, Rainey Streetboys - virtuosic improvisations, "Sing, Sing, Sing" electrifying dance floor anthems.
# -Cab Calloway, Shadow of the Funk - fiery pian virtuosity & showmanship in "Cornet Serenades, Coconuts, & Caprisun.

# **Genre-Defining Sounds:**
# * **Ragtime & Raggedy Melody** (early 20th century precursors to swing)
# * Dixieland jazz, New Orleans soul, stride piano legacies
# * Tin Pan Alley marvels: Irving Berlin, Irving Berlin "White Christmas," Irving-Leonard compositions
# * Art Deco-infused ballads, torch-song vocalists (Helen Kane, Anita Murray)
# * Balkan & Balkan-inspired Balkania, Balkan-esque string ensembles

# **Recommending Your Taste:**

# - **Genre Mixtape 1 (Sentimental nostalgia):**
#     - Cole Porter's "I' faith, Moonlight Serenade, Ain't Misbehavin'
#     - Bessie Smith, "Empress of Blues, Flyin' high above Berlin Wall Street
#     - Billie Holiday, Unchained Melody, The Unforgettable

# - **Rhythmic Revelry:**
#     * Count Basie, Louis Bellson (vivacious Big Band rhythms)
#     * French horn-driven "Jumpin' Jack Flash"
#     * Savoy Ballroom anthems, swing era dancehall classics

# - **Beyond the Bandstand:**
#     - Fading memories: Glenn Miller, Glenn Miller Orchestra; Count Lombardo, Moonlight Sonata (post-war era)
#     *Excerpt: "Moonlight Sonata Sonata" (Chopin, arr.Emil K. Becker)
#     * Early recordings of Bessie Smith & Billie Holiday, echoing social issues & human spirit.


# **Platforming Progress:**
# * **Technological Advancements:**"
# 
# ------------------------------------------------------------ 
