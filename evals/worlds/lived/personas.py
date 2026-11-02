"""The people of each simulated life. Life 1 (Jordan) is frozen: its world was grown from these tables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    text: str
    child: str
    spouse: str  # replaces "Jamie" in detail text
    work_company: str
    colliding_firsts: tuple[str, ...]
    shared_surname: str
    work: tuple[tuple[str, str], ...]
    clients: tuple[tuple[str, str, str], ...]
    recruiters: tuple[tuple[str, str, str], ...]
    family: tuple[tuple[str, str], ...]
    friends: tuple[tuple[str, str], ...]
    # (first, last, company, role): the role picks the service's own topic pool.
    services: tuple[tuple[str, str, str, str], ...]


JORDAN = Persona(
    name="jordan",
    text=(
        "Jordan Reyes, 34, senior product manager at Northwind Labs in Portland, Oregon. Married to Jamie, "
        "with a six-year-old daughter, Mia. Sister Elena and brother Mateo; cousins on the Doyle side (Mom's "
        "family) and the Alvarez side. Juggles a team, clients, recruiters, friends and family logistics "
        "through an AI assistant, mostly from their phone."
    ),
    child="Mia",
    spouse="Jamie",
    work_company="Northwind Labs",
    colliding_firsts=("Alice", "Ben"),
    shared_surname="Doyle",  # colleague Kevin Doyle and cousin Nora Doyle
    work=(
        ("Alice", "Chen"), ("Kevin", "Doyle"), ("Priya", "Shah"), ("Marcus", "Lee"), ("Hana", "Sato"),
        ("Omar", "Haddad"), ("Greta", "Fischer"), ("Ravi", "Iyer"), ("Chloe", "Martin"), ("Kenji", "Mori"),
        ("Zoe", "Turner"), ("Dmitri", "Petrov"), ("Maya", "Cohen"), ("Tobias", "Becker"), ("Sofia", "Rossi"),
        ("Dana", "Kim"), ("Felix", "Wagner"), ("Grace", "Liu"),
    ),
    clients=(
        ("Ben", "Carter", "Globex"), ("Hugo", "Lambert", "Initech"), ("Iris", "Novak", "Hooli"),
        ("Jonas", "Berg", "Umbrella"), ("Kira", "Nair", "Stark"), ("Leo", "Costa", "Wayne"),
        ("Petra", "Horvat", "Soylent"), ("Noah", "Fraser", "Acme"), ("Olga", "Ivanova", "Globex"),
        ("Pablo", "Ortega", "Initech"), ("Quinn", "Harper", "Hooli"), ("Seth", "Cole", "Umbrella"),
        ("Tara", "Singh", "Stark"), ("Yusuf", "Aydin", "Wayne"),
    ),
    recruiters=(
        ("Alice", "Brennan", "Hirewell"), ("Anjali", "Rao", "Talentbridge"), ("Gavin", "Hale", "Hirewell"),
        ("Wendy", "Zhou", "Pinnacle Search"), ("Leila", "Rahman", "Talentbridge"), ("Simon", "Abbott", "Pinnacle Search"),
    ),
    family=(
        ("Jamie", "Reyes"), ("Elena", "Reyes"), ("Mateo", "Reyes"), ("Nora", "Doyle"), ("Teresa", "Alvarez"),
        ("Luis", "Alvarez"), ("Diego", "Reyes"), ("Isabel", "Moreno"), ("Rafael", "Moreno"), ("Helen", "Walsh"),
        ("George", "Walsh"), ("Gloria", "Santos"), ("Pedro", "Alvarez"), ("Lucia", "Reyes"),
    ),
    friends=(
        ("Ben", "Okafor"), ("Rowan", "Brooks"), ("Beatriz", "Duarte"), ("Kwame", "Mensah"), ("Jake", "Morrison"),
        ("Freya", "Lindqvist"), ("Owen", "Ellis"), ("Fatima", "Malik"), ("Liam", "Byrne"), ("Callum", "Dunn"),
        ("Megan", "Hart"), ("Nikhil", "Bose"), ("Sara", "Lund"), ("Ethan", "Ward"), ("Julia", "Weber"),
        ("Marco", "Bianchi"), ("Anya", "Volkova"), ("Derek", "Shaw"),
    ),
    services=(
        ("Gary", "Whitman", "Whitman Properties", "landlord"), ("Laura", "Kemp", "Brightsmile Dental", "dentist"),
        ("Paul", "Fenwick", "Fenwick Tax", "accountant"), ("Rita", "Gomez", "Sparkle Home", "cleaner"),
        ("Steve", "Kowalski", "Eastside Auto", "mechanic"), ("Monica", "Reed", "Paws Clinic", "vet"),
        ("Henry", "Tran", "Tran Plumbing", "plumber"), ("Harriet", "Lawson", "Lawson Music", "piano teacher"),
        ("Carl", "Jensen", "Jensen Builders", "contractor"), ("Amy", "Fox", "Cascade Insurance", "insurance agent"),
    ),
)

RILEY = Persona(
    name="riley",
    text=(
        "Riley Nakamura, 41, operations director at Bluefin Logistics in Denver, Colorado. Partner Sam, "
        "two kids, Theo (9) and Lena (5). Sister Joanna and brother Kai; cousins on the Holm side (Mom's "
        "family) and the Sullivan side. Juggles a team, clients, recruiters, friends and family logistics "
        "through an AI assistant, mostly from their phone."
    ),
    child="Lena",
    spouse="Sam",
    work_company="Bluefin Logistics",
    colliding_firsts=("Nina", "Max"),
    shared_surname="Holm",  # colleague Ingrid Holm and cousins Erik and Ada Holm
    work=(
        ("Nina", "Castillo"), ("Victor", "Hayes"), ("Aisha", "Bello"), ("Bruno", "Silva"), ("Yuki", "Tanaka"),
        ("Farah", "Qureshi"), ("Lars", "Nilsson"), ("Mei", "Wong"), ("Adrian", "Popescu"), ("Ingrid", "Holm"),
        ("Samir", "Haddou"), ("Clara", "Dubois"), ("Rohan", "Mehta"), ("Elise", "Moreau"), ("Tomasz", "Nowak"),
        ("Imani", "Okeke"), ("Hector", "Ruiz"), ("Signe", "Dahl"),
    ),
    clients=(
        ("Max", "Weller", "Globex"), ("Odette", "Fournier", "Initech"), ("Raj", "Pillai", "Hooli"),
        ("Bianca", "Ferreira", "Umbrella"), ("Caleb", "Young", "Stark"), ("Delia", "Murray", "Wayne"),
        ("Emeka", "Obi", "Soylent"), ("Fiona", "Grant", "Acme"), ("Gustavo", "Lima", "Globex"),
        ("Heidi", "Kraus", "Initech"), ("Ivan", "Petrenko", "Hooli"), ("Jade", "Thornton", "Umbrella"),
        ("Kofi", "Asante", "Stark"), ("Lorna", "Price", "Wayne"),
    ),
    recruiters=(
        ("Mona", "Ellison", "Hirewell"), ("Nate", "Rivers", "Talentbridge"), ("Opal", "Grey", "Hirewell"),
        ("Pierce", "Lang", "Pinnacle Search"), ("Rosa", "Vidal", "Talentbridge"), ("Stefan", "Kroll", "Pinnacle Search"),
    ),
    family=(
        ("Sam", "Nakamura"), ("Joanna", "Nakamura"), ("Kai", "Nakamura"), ("Erik", "Holm"), ("Martha", "Nakamura"),
        ("Hiro", "Nakamura"), ("Ada", "Holm"), ("Bea", "Sullivan"), ("Frank", "Sullivan"), ("Tessa", "Nakamura"),
        ("Ollie", "Sullivan"), ("June", "Park"), ("Walter", "Nakamura"), ("Ruth", "Sullivan"),
    ),
    friends=(
        ("Nina", "Park"), ("Max", "Okonkwo"), ("Zara", "Ahmed"), ("Brendan", "Kelly"), ("Lucy", "Hammond"),
        ("Tariq", "Aziz"), ("Vera", "Kowal"), ("Will", "Danvers"), ("Ximena", "Rojas"), ("Yara", "Saleh"),
        ("Abby", "Wren"), ("Cole", "Mercer"), ("Dina", "Farouk"), ("Eli", "Strand"), ("Flora", "Bayer"),
        ("Gus", "Moran"), ("Holly", "Pratt"), ("Ian", "Doherty"),
    ),
    services=(
        ("Dennis", "Varga", "Varga Rentals", "landlord"), ("Paula", "Ng", "Bright Day Dental", "dentist"),
        ("Arthur", "Bell", "Bell Tax", "accountant"), ("Marisol", "Vega", "Fresh Nest", "cleaner"),
        ("Bob", "Kaminski", "Mile High Auto", "mechanic"), ("Nadia", "Brandt", "Cherry Creek Vets", "vet"),
        ("Ray", "Duffy", "Duffy Plumbing", "plumber"), ("Edith", "Lowe", "Lowe Music", "piano teacher"),
        ("Trent", "Hollis", "Hollis Build", "contractor"), ("Wanda", "Mills", "Summit Insurance", "insurance agent"),
    ),
)

PERSONAS = (JORDAN, RILEY)


def persona_for(seed: int) -> Persona:
    """Seed 1 is life 1 (Jordan, frozen); every other seed is life 2 (Riley)."""
    return JORDAN if seed == 1 else RILEY
