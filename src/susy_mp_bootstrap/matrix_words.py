from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


X = "X"
P = "P"
PSI = "PSI"
PSIDAG = "PSIDAG"

LEVELS = {
    X: Fraction(1, 1),
    P: Fraction(2, 1),
    PSI: Fraction(3, 2),
    PSIDAG: Fraction(3, 2),
}

DAGGER = {
    X: X,
    P: P,
    PSI: PSIDAG,
    PSIDAG: PSI,
}


@dataclass(frozen=True, order=True)
class TracedWord:
    letters: tuple[str, ...] = ()

    @classmethod
    def from_letters(cls, *letters: str) -> "TracedWord":
        return cls(tuple(letters))

    def level(self) -> Fraction:
        return sum((LEVELS[letter] for letter in self.letters), start=Fraction(0, 1))

    def fermion_number(self) -> int:
        return sum(1 if letter == PSIDAG else -1 if letter == PSI else 0 for letter in self.letters)

    def dagger(self) -> "TracedWord":
        return TracedWord(tuple(DAGGER[letter] for letter in reversed(self.letters)))

    def cyclic_canonical(self) -> "TracedWord":
        if not self.letters:
            return self
        rotations = [self.letters[offset:] + self.letters[:offset] for offset in range(len(self.letters))]
        return TracedWord(min(rotations))

    def reality_sign(self) -> int:
        return -1 if sum(1 for letter in self.letters if letter == P) % 2 else 1

