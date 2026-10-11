/**
 * Builds the plain-text citation and BibTeX entry shown in the Citation
 * section of a paper page. Used by PaperPost.astro when a paper sets
 * `cite: true` and does not provide a manual `citation` override.
 */

export interface CitationSource {
    authors: string[];
    title: string;
    pubDate: Date;
    venue?: string;
    url: string;
}

const MONTHS_SHORT = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];
const MONTHS_LONG = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const STOPWORDS = new Set(['a', 'an', 'the', 'on', 'of', 'in', 'and', 'for', 'to', 'with', 'from']);

/** Splits "First Middle Last" into its first names and family name. */
function splitName(name: string): { first: string; last: string } {
    const parts = name.trim().split(/\s+/);
    if (parts.length < 2) return { first: '', last: parts[0] ?? name };
    return { first: parts.slice(0, -1).join(' '), last: parts[parts.length - 1] };
}

function bibtexAuthor(name: string): string {
    const { first, last } = splitName(name);
    return first ? `${last}, ${first}` : last;
}

/** Authors as they appear in the plain-text citation, e.g. "Fleming, James, et al." */
function plainAuthors(authors: string[]): string {
    if (authors.length === 0) return '';
    const first = bibtexAuthor(authors[0]);
    switch (authors.length) {
        case 1:
            return `${first}.`;
        case 2:
            return `${first}, and ${authors[1]}.`;
        case 3:
            return `${first}, ${authors[1]}, and ${authors[2]}.`;
        default:
            return `${first}, et al.`;
    }
}

/** BibTeX key like "fleming2026sample" (family name + year + first title word). */
function citationKey(authors: string[], title: string, year: number): string {
    const family = authors[0]
        ? splitName(authors[0]).last.toLowerCase().replace(/[^a-z0-9]/g, '')
        : 'anonymous';
    const words = title.toLowerCase().match(/[a-z0-9]+/g) ?? [];
    const titleWord = words.find((word) => !STOPWORDS.has(word)) ?? 'paper';
    return `${family}${year}${titleWord}`;
}

export function generateCitation({ authors, title, pubDate, venue, url }: CitationSource): {
    text: string;
    bibtex: string;
} {
    const year = pubDate.getFullYear();
    const monthLong = MONTHS_LONG[pubDate.getMonth()];
    const monthShort = MONTHS_SHORT[pubDate.getMonth()];

    const authorPrefix = plainAuthors(authors);
    const venuePrefix = venue ? `${venue}, ` : '';
    const text = `${authorPrefix}${authorPrefix ? ' ' : ''}“${title}.” ${venuePrefix}${monthLong} ${year}.`;

    const fields = [
        ...(authors.length > 0 ? [`author = {${authors.map(bibtexAuthor).join(' and ')}}`] : []),
        `title = {{${title}}}`,
        `year = {${year}}`,
        `month = ${monthShort}`,
        ...(venue ? [`howpublished = {${venue}}`] : []),
        `url = {${url}}`,
    ];

    const bibtex = `@misc{${citationKey(authors, title, year)},\n  ${fields.join(',\n  ')}\n}`;

    return { text, bibtex };
}
