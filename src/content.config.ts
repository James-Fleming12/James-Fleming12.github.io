import { defineCollection } from 'astro:content';
import { z } from 'astro:schema';
import { glob } from 'astro/loaders';

const blog = defineCollection({
    loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
    // Type-check frontmatter using a schema
    schema: z.object({
        title: z.string(),
        description: z.string(),
        // Transform string to Date object
        pubDate: z.coerce.date(),
        updatedDate: z.coerce.date().optional(),
        heroImage: z.string().optional(),
    }),
});

const papers = defineCollection({
    loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/papers' }),
    // Type-check frontmatter using a schema
    schema: z.object({
        title: z.string(),
        description: z.string(),
        authors: z.array(z.string()),
        // Transform string to Date object
        pubDate: z.coerce.date(),
        updatedDate: z.coerce.date().optional(),
        // Journal, conference or preprint venue, e.g. "NeurIPS 2026"
        venue: z.string().optional(),
        // Canonical URL of the paper, used in generated citations for papers
        // hosted elsewhere. Defaults to the paper's page on this site.
        url: z.string().url().optional(),
        // Path to a PDF hosted on this site (under public/), shown in an embedded viewer.
        // Omit for papers hosted elsewhere and use `links` instead.
        pdf: z.string().optional(),
        // External links to the paper (arXiv, IEEE Xplore, DOI, ...)
        links: z
            .array(
                z.object({
                    label: z.string(),
                    url: z.string().url(),
                }),
            )
            .optional(),
        // Show an AI disclaimer before the abstract, for papers written with AI
        // rather than by the author.
        ai: z.boolean().default(false),
        // Show a citation section on the paper's page. When true, the citation is
        // generated automatically from the fields above, unless a manual
        // `citation` is provided below. Leave unset/false for pieces that should
        // not be cited (e.g. AI rewrites or simplifications).
        cite: z.boolean().default(false),
        // Optional manual citation that overrides the generated one.
        citation: z
            .object({
                text: z.string(),
                bibtex: z.string(),
            })
            .optional(),
    }),
});

export const collections = { blog, papers };
